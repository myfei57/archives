package archives_test

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	stdzip "archive/zip"

	kzip "github.com/klauspost/compress/zip"
	"github.com/mholt/archives"
)

// ---------- test fixtures and helpers ----------

type zipEntrySpec struct {
	name          string
	body          []byte
	dir           bool
	symlinkTarget string
}

// buildZip writes a ZIP to memory preserving the given entry order so tests
// can construct archives whose children precede their parent directories.
func buildZip(t *testing.T, specs []zipEntrySpec) []byte {
	t.Helper()
	var buf bytes.Buffer
	zw := stdzip.NewWriter(&buf)
	for _, spec := range specs {
		hdr := &stdzip.FileHeader{Name: spec.name}
		switch {
		case spec.dir:
			if !strings.HasSuffix(hdr.Name, "/") {
				hdr.Name += "/"
			}
			hdr.Method = stdzip.Store
		case spec.symlinkTarget != "":
			hdr.Method = stdzip.Store
			hdr.SetMode(os.ModeSymlink | 0o777)
		default:
			hdr.Method = stdzip.Deflate
		}
		writer, err := zw.CreateHeader(hdr)
		if err != nil {
			t.Fatalf("creating header %q: %v", spec.name, err)
		}
		if spec.dir {
			continue
		}
		payload := spec.body
		if spec.symlinkTarget != "" {
			payload = []byte(spec.symlinkTarget)
		}
		if _, err := writer.Write(payload); err != nil {
			t.Fatalf("writing body %q: %v", spec.name, err)
		}
	}
	if err := zw.Close(); err != nil {
		t.Fatalf("closing zip writer: %v", err)
	}
	return buf.Bytes()
}

// trackingSource wraps bytes.Reader to observe concurrent ReaderAt usage,
// seek calls, and any attempt by the framework to close the caller's source.
type trackingSource struct {
	*bytes.Reader
	// delay, when positive, is held for the duration of each ReadAt so tests
	// can deterministically observe overlapping reads on the shared source.
	delay time.Duration

	mu           sync.Mutex
	readAtActive int
	readAtPeak   int
	readAtCalls  int
	seekCalls    int
	closeCalls   int
}

func newTrackingSource(data []byte) *trackingSource {
	return &trackingSource{Reader: bytes.NewReader(data)}
}

func (s *trackingSource) ReadAt(p []byte, off int64) (int, error) {
	s.mu.Lock()
	s.readAtActive++
	s.readAtCalls++
	if s.readAtActive > s.readAtPeak {
		s.readAtPeak = s.readAtActive
	}
	s.mu.Unlock()

	if s.delay > 0 {
		time.Sleep(s.delay)
	}
	n, err := s.Reader.ReadAt(p, off)

	s.mu.Lock()
	s.readAtActive--
	s.mu.Unlock()
	return n, err
}

func (s *trackingSource) Seek(offset int64, whence int) (int64, error) {
	s.mu.Lock()
	s.seekCalls++
	s.mu.Unlock()
	return s.Reader.Seek(offset, whence)
}

func (s *trackingSource) Close() error {
	s.mu.Lock()
	s.closeCalls++
	s.mu.Unlock()
	return nil
}

func (s *trackingSource) snapshot() (calls, peak, seeks, closes int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.readAtCalls, s.readAtPeak, s.seekCalls, s.closeCalls
}

// eventLog records handler start/completion order and concurrency peaks.
type eventLog struct {
	mu        sync.Mutex
	started   []string
	completed []string
	active    int
	peak      int
}

func (l *eventLog) enter(name string) func() {
	l.mu.Lock()
	l.started = append(l.started, name)
	l.active++
	if l.active > l.peak {
		l.peak = l.active
	}
	l.mu.Unlock()
	return func() {
		l.mu.Lock()
		l.completed = append(l.completed, name)
		l.active--
		l.mu.Unlock()
	}
}

func (l *eventLog) startIndex(name string) int {
	l.mu.Lock()
	defer l.mu.Unlock()
	for i, candidate := range l.started {
		if candidate == name {
			return i
		}
	}
	return -1
}

func (l *eventLog) startedNames() map[string]bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	names := make(map[string]bool, len(l.started))
	for _, name := range l.started {
		names[name] = true
	}
	return names
}

func (l *eventLog) concurrencyPeak() int {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.peak
}

// waitUntil polls cond until it is true or the deadline elapses.
func waitUntil(t *testing.T, timeout time.Duration, cond func() bool, message string) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(2 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for condition: %s", message)
}

func testContext(t *testing.T) (context.Context, context.CancelFunc) {
	return context.WithTimeout(context.Background(), 30*time.Second)
}

// assertEveryOrdinalReported makes sure no ordinal was silently dropped.
func assertEveryOrdinalReported(t *testing.T, result archives.ExtractParallelResult, expected int) {
	t.Helper()
	if len(result.Reports) != expected {
		t.Fatalf("expected %d reports, got %d", expected, len(result.Reports))
	}
	for ordinal, report := range result.Reports {
		if report.Ordinal != ordinal {
			t.Fatalf("report at index %d has ordinal %d; reports must follow original order", ordinal, report.Ordinal)
		}
	}
}

// readAllContent opens and fully reads a regular entry through the handler.
func readAllContent(t *testing.T, file archives.FileInfo) []byte {
	t.Helper()
	handle, err := file.Open()
	if err != nil {
		t.Fatalf("opening %q: %v", file.NameInArchive, err)
	}
	defer handle.Close()
	data, err := io.ReadAll(handle)
	if err != nil {
		t.Fatalf("reading %q: %v", file.NameInArchive, err)
	}
	return data
}

// ---------- tests ----------

func TestZipParallel_MultiDirContentAndOrdinalReports(t *testing.T) {
	specs := []zipEntrySpec{
		{name: "top.txt", body: []byte("top")},
		{name: "a/", dir: true},
		{name: "a/x.txt", body: []byte("x-content")},
		{name: "a/sub/", dir: true},
		{name: "a/sub/y.txt", body: []byte("y-content-longer")},
		{name: "b/", dir: true},
		{name: "b/z.txt", body: []byte("z")},
	}
	source := newTrackingSource(buildZip(t, specs))

	z := archives.Zip{}
	ctx, cancel := testContext(t)
	defer cancel()

	var (
		gotMu sync.Mutex
		got   = make(map[string]string)
	)
	result, err := z.ExtractParallel(ctx, source, archives.ExtractParallelOptions{
		MaxWorkers: 4, MaxOpenReaders: 3,
	}, func(ctx context.Context, file archives.FileInfo) error {
		if file.IsDir() {
			return nil
		}
		content := string(readAllContent(t, file))
		gotMu.Lock()
		got[file.NameInArchive] = content
		gotMu.Unlock()
		return nil
	})
	if err != nil {
		t.Fatalf("ExtractParallel failed: %v", err)
	}
	assertEveryOrdinalReported(t, result, len(specs))

	want := map[string]string{
		"top.txt":     "top",
		"a/x.txt":     "x-content",
		"a/sub/y.txt": "y-content-longer",
		"b/z.txt":     "z",
	}
	if len(got) != len(want) {
		t.Fatalf("handled %d files, want %d", len(got), len(want))
	}
	for name, body := range want {
		if got[name] != body {
			t.Errorf("content of %q = %q, want %q", name, got[name], body)
		}
	}
	for _, report := range result.Reports {
		if report.Outcome != archives.ParallelEntryCompleted {
			t.Errorf("entry %q outcome %v with err %v, want completed", report.NameInArchive, report.Outcome, report.Err)
		}
	}
	// Byte accounting must reflect real reads of the regular files.
	for _, report := range result.Reports {
		if report.IsDir {
			if report.BytesRead != 0 {
				t.Errorf("directory %q billed %d bytes", report.NameInArchive, report.BytesRead)
			}
			continue
		}
		if report.BytesRead != int64(len(want[report.NameInArchive])) {
			t.Errorf("entry %q billed %d bytes, want %d", report.NameInArchive, report.BytesRead, len(want[report.NameInArchive]))
		}
	}
	if _, _, _, closes := source.snapshot(); closes != 0 {
		t.Errorf("framework closed the caller's source %d times", closes)
	}
}

func TestZipParallel_UnorderedTestDataDirectoryFirst(t *testing.T) {
	raw, err := os.ReadFile("testdata/unordered.zip")
	if err != nil {
		t.Fatalf("reading testdata: %v", err)
	}
	source := newTrackingSource(raw)

	var events eventLog
	z := archives.Zip{}
	ctx, cancel := testContext(t)
	defer cancel()

	result, err := z.ExtractParallel(ctx, source, archives.ExtractParallelOptions{MaxWorkers: 3},
		func(ctx context.Context, file archives.FileInfo) error {
			done := events.enter(file.NameInArchive)
			defer done()
			return nil
		})
	if err != nil {
		t.Fatalf("ExtractParallel failed: %v", err)
	}
	assertEveryOrdinalReported(t, result, 5)

	// Even though "2/1" is listed before "1/2" in the archive, every
	// directory handler must start before any of its descendants.
	for _, pair := range [][2]string{{"1/", "1/1"}, {"1/", "1/2"}, {"2/", "2/1"}} {
		dirIdx := events.startIndex(pair[0])
		childIdx := events.startIndex(pair[1])
		if dirIdx < 0 || childIdx < 0 {
			t.Fatalf("missing start event for %v", pair)
		}
		if dirIdx > childIdx {
			t.Errorf("directory %q started at %d after child %q at %d", pair[0], dirIdx, pair[1], childIdx)
		}
	}
}

func TestZipParallel_ChildrenQueuedBeforeDirectoryRespectSkipDir(t *testing.T) {
	// Children are physically written before the directory entry.
	specs := []zipEntrySpec{
		{name: "d/a.txt", body: []byte("a")},
		{name: "d/b.txt", body: []byte("bb")},
		{name: "d/", dir: true},
		{name: "outside.txt", body: []byte("out")},
	}
	data := buildZip(t, specs)

	t.Run("directory handler starts first", func(t *testing.T) {
		var events eventLog
		z := archives.Zip{}
		ctx, cancel := testContext(t)
		defer cancel()

		if _, err := z.ExtractParallel(ctx, newTrackingSource(data),
			archives.ExtractParallelOptions{MaxWorkers: 4},
			func(ctx context.Context, file archives.FileInfo) error {
				done := events.enter(file.NameInArchive)
				defer done()
				return nil
			}); err != nil {
			t.Fatalf("ExtractParallel failed: %v", err)
		}
		dirIdx := events.startIndex("d/")
		for _, child := range []string{"d/a.txt", "d/b.txt"} {
			childIdx := events.startIndex(child)
			if dirIdx < 0 || childIdx < 0 || dirIdx > childIdx {
				t.Errorf("directory event %d must precede child %q event %d", dirIdx, child, childIdx)
			}
		}
	})

	t.Run("SkipDir excludes descendants regardless of entry order", func(t *testing.T) {
		var events eventLog
		z := archives.Zip{}
		ctx, cancel := testContext(t)
		defer cancel()

		result, err := z.ExtractParallel(ctx, newTrackingSource(data),
			archives.ExtractParallelOptions{MaxWorkers: 4},
			func(ctx context.Context, file archives.FileInfo) error {
				done := events.enter(file.NameInArchive)
				defer done()
				if file.IsDir() && file.NameInArchive == "d/" {
					return fs.SkipDir
				}
				return nil
			})
		if err != nil {
			t.Fatalf("ExtractParallel failed: %v", err)
		}
		assertEveryOrdinalReported(t, result, len(specs))

		names := events.startedNames()
		if names["d/a.txt"] || names["d/b.txt"] {
			t.Errorf("skipped descendants were handed to a handler: %v", names)
		}
		if !names["d/"] || !names["outside.txt"] {
			t.Errorf("directory and outside entry must still run: %v", names)
		}
		byName := make(map[string]archives.ParallelEntryReport)
		for _, report := range result.Reports {
			byName[report.NameInArchive] = report
		}
		for _, child := range []string{"d/a.txt", "d/b.txt"} {
			if byName[child].Outcome != archives.ParallelEntrySkipped {
				t.Errorf("child %q outcome = %v, want skipped", child, byName[child].Outcome)
			}
		}
		if byName["outside.txt"].Outcome != archives.ParallelEntryCompleted {
			t.Errorf("outside.txt outcome = %v, want completed", byName["outside.txt"].Outcome)
		}
	})
}

func TestZipParallel_RealConcurrencyAndReaderPeak(t *testing.T) {
	const (
		workers    = 4
		maxReaders = 2
		fileCount  = 8
		bodySize   = 4096
	)
	specs := make([]zipEntrySpec, fileCount)
	for i := range specs {
		specs[i] = zipEntrySpec{
			name: fmt.Sprintf("f-%02d.bin", i),
			body: bytes.Repeat([]byte{byte('a' + i%26)}, bodySize),
		}
	}
	source := newTrackingSource(buildZip(t, specs))
	source.delay = 2 * time.Millisecond

	var (
		events     eventLog
		openActive atomic.Int32
		openPeak   atomic.Int32
		release    = make(chan struct{})
	)

	z := archives.Zip{}
	ctx, cancel := testContext(t)
	defer cancel()

	done := make(chan error, 1)
	go func() {
		_, err := z.ExtractParallel(ctx, source, archives.ExtractParallelOptions{
			MaxWorkers: workers, MaxOpenReaders: maxReaders,
		}, func(ctx context.Context, file archives.FileInfo) error {
			leave := events.enter(file.NameInArchive)
			defer leave()

			handle, err := file.Open()
			if err != nil {
				return err
			}
			cur := openActive.Add(1)
			defer openActive.Add(-1)
			for {
				peak := openPeak.Load()
				if cur <= peak || openPeak.CompareAndSwap(peak, cur) {
					break
				}
			}
			// Event barrier: hold the reader until the test releases.
			select {
			case <-release:
			case <-ctx.Done():
				handle.Close()
				return ctx.Err()
			}
			if _, err := io.Copy(io.Discard, handle); err != nil {
				handle.Close()
				return err
			}
			return handle.Close()
		})
		done <- err
	}()

	// Four handlers must be active concurrently, but only two readers open.
	waitUntil(t, 5*time.Second, func() bool { return events.concurrencyPeak() >= workers },
		"worker concurrency peak")
	waitUntil(t, 5*time.Second, func() bool { return openPeak.Load() >= maxReaders },
		"open-reader peak")
	// Give the barrier a moment to prove the reader cap is never exceeded.
	time.Sleep(50 * time.Millisecond)
	if peak := openPeak.Load(); peak > maxReaders {
		t.Fatalf("open reader peak %d exceeds hard limit %d", peak, maxReaders)
	}
	if peak := events.concurrencyPeak(); peak > workers {
		t.Fatalf("handler peak %d exceeds MaxWorkers %d", peak, workers)
	}
	close(release)

	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("ExtractParallel failed: %v", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("ExtractParallel did not finish after barrier release")
	}
	if peak := events.concurrencyPeak(); peak != workers {
		t.Errorf("handler concurrency peak = %d, want exactly %d", peak, workers)
	}
	if peak := openPeak.Load(); peak != maxReaders {
		t.Errorf("open-reader peak = %d, want exactly %d", peak, maxReaders)
	}
	_, readAtPeak, _, closes := source.snapshot()
	if readAtPeak < 2 {
		t.Errorf("expected concurrent ReadAt calls on the shared source, peak = %d", readAtPeak)
	}
	if closes != 0 {
		t.Errorf("framework closed the caller's source %d times", closes)
	}
}

func TestZipParallel_CancelStopsSlowHandlers(t *testing.T) {
	const fileCount = 6
	specs := make([]zipEntrySpec, fileCount)
	for i := range specs {
		specs[i] = zipEntrySpec{name: fmt.Sprintf("f-%02d.txt", i), body: []byte("data")}
	}
	source := newTrackingSource(buildZip(t, specs))

	var active atomic.Int32
	z := archives.Zip{}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)

	finished := make(chan struct{})
	go func() {
		defer close(finished)
		result, err := z.ExtractParallel(ctx, source,
			archives.ExtractParallelOptions{MaxWorkers: 2, MaxOpenReaders: 2},
			func(ctx context.Context, file archives.FileInfo) error {
				active.Add(1)
				defer active.Add(-1)
				// Cooperative blocking handler: only the context can release it.
				<-ctx.Done()
				return ctx.Err()
			})
		assertEveryOrdinalReported(t, result, fileCount)
		if !errors.Is(err, context.Canceled) {
			t.Errorf("expected context.Canceled, got %v", err)
		}
		var notRun int
		for _, report := range result.Reports {
			if report.Outcome == archives.ParallelEntryNotRun {
				notRun++
			}
		}
		if notRun == 0 {
			t.Error("expected at least one entry to be reported as not-run after cancellation")
		}
	}()

	waitUntil(t, 5*time.Second, func() bool { return active.Load() >= 2 }, "workers to block")
	cancel()

	select {
	case <-finished:
	case <-time.After(10 * time.Second):
		t.Fatal("ExtractParallel did not return after context cancellation")
	}
	if _, _, _, closes := source.snapshot(); closes != 0 {
		t.Errorf("framework closed the caller's source %d times", closes)
	}
}

func TestZipParallel_SkipAllStopsSchedulingAndCancelsInFlight(t *testing.T) {
	const fileCount = 6
	specs := make([]zipEntrySpec, fileCount)
	for i := range specs {
		specs[i] = zipEntrySpec{name: fmt.Sprintf("f-%03d.txt", i), body: []byte("data")}
	}

	var (
		ctxCanceledObserved atomic.Bool
		events              eventLog
	)
	z := archives.Zip{}
	ctx, cancel := testContext(t)
	defer cancel()

	result, err := z.ExtractParallel(ctx, newTrackingSource(buildZip(t, specs)),
		archives.ExtractParallelOptions{MaxWorkers: 3},
		func(ctx context.Context, file archives.FileInfo) error {
			done := events.enter(file.NameInArchive)
			defer done()

			if file.NameInArchive == "f-002.txt" {
				return fs.SkipAll
			}
			// Other in-flight handlers block until the SkipAll cancellation.
			select {
			case <-ctx.Done():
				ctxCanceledObserved.Store(true)
				return nil // cooperative wind-down, not a handler failure
			case <-time.After(5 * time.Second):
				return errors.New("handler was not canceled in time")
			}
		})
	if err != nil {
		t.Fatalf("SkipAll must not surface as an error, got: %v", err)
	}
	assertEveryOrdinalReported(t, result, fileCount)
	if !ctxCanceledObserved.Load() {
		t.Error("in-flight handlers did not observe context cancellation after SkipAll")
	}

	byName := make(map[string]archives.ParallelEntryReport)
	for _, report := range result.Reports {
		byName[report.NameInArchive] = report
	}
	if byName["f-002.txt"].Outcome != archives.ParallelEntryCompleted {
		t.Errorf("SkipAll entry outcome = %v, want completed", byName["f-002.txt"].Outcome)
	}
	for i := 3; i < fileCount; i++ {
		name := fmt.Sprintf("f-%03d.txt", i)
		if byName[name].Outcome != archives.ParallelEntryNotRun {
			t.Errorf("entry %q outcome = %v, want not-run after SkipAll", name, byName[name].Outcome)
		}
		if byName[name].Err != nil {
			t.Errorf("not-run entry %q carries error %v", name, byName[name].Err)
		}
	}
	for _, report := range result.Reports {
		if report.Outcome == archives.ParallelEntryFailed {
			t.Errorf("no entry may fail after clean SkipAll; %q failed: %v", report.NameInArchive, report.Err)
		}
	}
}

func TestZipParallel_CRCFailureIsReportedEvenWhenHandlerIgnoresReads(t *testing.T) {
	specs := []zipEntrySpec{
		{name: "good.txt", body: []byte("good contents")},
		{name: "stored.bin", body: bytes.Repeat([]byte("Z"), 256)},
	}
	data := buildZip(t, specs)

	// Rewrite stored.bin uncompressed, then flip one payload byte so its CRC fails.
	data = rewriteStored(t, data, "stored.bin")
	offset := storedDataOffset(t, data, "stored.bin")
	data[offset+1] ^= 0xFF

	z := archives.Zip{}
	ctx, cancel := testContext(t)
	defer cancel()

	result, err := z.ExtractParallel(ctx, newTrackingSource(data),
		archives.ExtractParallelOptions{MaxWorkers: 2},
		func(ctx context.Context, file archives.FileInfo) error {
			if file.IsDir() {
				return nil
			}
			handle, openErr := file.Open()
			if openErr != nil {
				return openErr
			}
			_, _ = io.Copy(io.Discard, handle) // deliberately ignore the read error
			return handle.Close()
		})
	if err == nil {
		t.Fatal("expected an aggregated CRC error, got nil")
	}
	if !errors.Is(err, kzip.ErrChecksum) {
		t.Fatalf("expected errors.Is(err, zip.ErrChecksum), got %v", err)
	}
	assertEveryOrdinalReported(t, result, len(specs))

	byName := make(map[string]archives.ParallelEntryReport)
	for _, report := range result.Reports {
		byName[report.NameInArchive] = report
	}
	corrupt := byName["stored.bin"]
	if corrupt.Outcome != archives.ParallelEntryFailed || !errors.Is(corrupt.Err, kzip.ErrChecksum) {
		t.Fatalf("corrupt entry report = %+v, want failed with checksum error", corrupt)
	}
	if byName["good.txt"].Outcome != archives.ParallelEntryCompleted {
		t.Errorf("good entry outcome = %v, want completed", byName["good.txt"].Outcome)
	}
}

func TestZipParallel_PerEntryBudget(t *testing.T) {
	specs := []zipEntrySpec{
		{name: "big.bin", body: bytes.Repeat([]byte("x"), 1000)},
		{name: "small.bin", body: []byte("tiny")},
	}

	t.Run("ContinueOnError aggregates and keeps going", func(t *testing.T) {
		z := archives.Zip{ContinueOnError: true}
		ctx, cancel := testContext(t)
		defer cancel()

		result, err := z.ExtractParallel(ctx, newTrackingSource(buildZip(t, specs)),
			archives.ExtractParallelOptions{MaxWorkers: 2, MaxEntryBytes: 100},
			func(ctx context.Context, file archives.FileInfo) error {
				handle, openErr := file.Open()
				if openErr != nil {
					return openErr
				}
				defer handle.Close()
				_, _ = io.Copy(io.Discard, handle)
				return nil
			})
		if err == nil || !errors.Is(err, archives.ErrEntrySizeLimit) {
			t.Fatalf("expected ErrEntrySizeLimit in returned error, got %v", err)
		}
		assertEveryOrdinalReported(t, result, len(specs))
		byName := reportsByName(result)
		big := byName["big.bin"]
		if big.Outcome != archives.ParallelEntryFailed || !errors.Is(big.Err, archives.ErrEntrySizeLimit) {
			t.Fatalf("big entry report = %+v, want entry-limit failure", big)
		}
		if big.BytesRead <= 100 {
			t.Errorf("big entry billed %d bytes, expected real reads past the limit", big.BytesRead)
		}
		small := byName["small.bin"]
		if small.Outcome != archives.ParallelEntryCompleted {
			t.Errorf("small entry outcome = %v, want completed", small.Outcome)
		}
	})

	t.Run("stop mode treats the breach as terminal", func(t *testing.T) {
		z := archives.Zip{}
		ctx, cancel := testContext(t)
		defer cancel()

		result, err := z.ExtractParallel(ctx, newTrackingSource(buildZip(t, specs)),
			archives.ExtractParallelOptions{MaxWorkers: 1, MaxEntryBytes: 100},
			func(ctx context.Context, file archives.FileInfo) error {
				handle, openErr := file.Open()
				if openErr != nil {
					return openErr
				}
				defer handle.Close()
				_, _ = io.Copy(io.Discard, handle)
				return nil
			})
		if !errors.Is(err, archives.ErrEntrySizeLimit) {
			t.Fatalf("expected ErrEntrySizeLimit, got %v", err)
		}
		assertEveryOrdinalReported(t, result, len(specs))
		if reportsByName(result)["small.bin"].Outcome != archives.ParallelEntryNotRun {
			t.Errorf("small entry outcome = %v, want not-run after terminal breach",
				reportsByName(result)["small.bin"].Outcome)
		}
	})
}

func TestZipParallel_TotalBudgetIsTerminalEvenWithContinueOnError(t *testing.T) {
	const fileCount, fileSize = 5, 100
	specs := make([]zipEntrySpec, fileCount)
	for i := range specs {
		specs[i] = zipEntrySpec{
			name: fmt.Sprintf("f-%02d.bin", i),
			body: bytes.Repeat([]byte{byte(i)}, fileSize),
		}
	}
	z := archives.Zip{ContinueOnError: true}
	ctx, cancel := testContext(t)
	defer cancel()

	result, err := z.ExtractParallel(ctx, newTrackingSource(buildZip(t, specs)),
		archives.ExtractParallelOptions{
			MaxWorkers:           2,
			MaxDecompressedBytes: 250,
		},
		func(ctx context.Context, file archives.FileInfo) error {
			handle, openErr := file.Open()
			if openErr != nil {
				return openErr
			}
			defer handle.Close()
			_, _ = io.Copy(io.Discard, handle)
			return nil
		})
	if err == nil || !errors.Is(err, archives.ErrTotalSizeLimit) {
		t.Fatalf("ContinueOnError must not hide the total budget breach, got %v", err)
	}
	assertEveryOrdinalReported(t, result, fileCount)

	var billed int64
	var completed, notRun, failed int
	var sawTotalLimitError bool
	for _, report := range result.Reports {
		billed += report.BytesRead
		switch report.Outcome {
		case archives.ParallelEntryCompleted:
			completed++
		case archives.ParallelEntryNotRun:
			notRun++
			if report.BytesRead != 0 {
				t.Errorf("not-run entry %q billed %d bytes", report.NameInArchive, report.BytesRead)
			}
		case archives.ParallelEntryFailed:
			failed++
			// The crossing entry carries the budget error; other in-flight
			// entries may only see the resulting context cancellation.
			if !errors.Is(report.Err, archives.ErrTotalSizeLimit) &&
				!errors.Is(report.Err, context.Canceled) {
				t.Errorf("failed entry %q err = %v, want total-limit or cancellation error",
					report.NameInArchive, report.Err)
			}
			if errors.Is(report.Err, archives.ErrTotalSizeLimit) {
				sawTotalLimitError = true
			}
		}
	}
	if !sawTotalLimitError {
		t.Error("at least one report must carry ErrTotalSizeLimit")
	}
	if notRun == 0 {
		t.Error("expected entries that never started after the total budget was exhausted")
	}
	if completed >= fileCount {
		t.Error("not all entries may complete once the total budget is exhausted")
	}
	if billed < 251 || billed > fileCount*fileSize {
		t.Errorf("billed %d bytes, expected real billing just past the 250 limit", billed)
	}
}

func TestZipParallel_PathConflicts(t *testing.T) {
	t.Run("fail policy stops at planning before any handler", func(t *testing.T) {
		specs := []zipEntrySpec{
			{name: "a.txt", body: []byte("first")},
			{name: "a.txt", body: []byte("second")},
			{name: "x", body: []byte("file-as-dir")},
			{name: "x/child.txt", body: []byte("child")},
		}
		var calls atomic.Int32
		z := archives.Zip{}
		ctx, cancel := testContext(t)
		defer cancel()

		_, err := z.ExtractParallel(ctx, newTrackingSource(buildZip(t, specs)),
			archives.ExtractParallelOptions{MaxWorkers: 2, ConflictPolicy: archives.ZipConflictFail},
			func(ctx context.Context, file archives.FileInfo) error {
				calls.Add(1)
				return nil
			})
		if !errors.Is(err, archives.ErrPathConflict) {
			t.Fatalf("expected ErrPathConflict, got %v", err)
		}
		if calls.Load() != 0 {
			t.Fatalf("planning conflict must prevent handler calls, got %d", calls.Load())
		}
	})

	t.Run("skip policy marks later ordinals and extracts the rest", func(t *testing.T) {
		specs := []zipEntrySpec{
			{name: "a.txt", body: []byte("first")},
			{name: "a.txt", body: []byte("second")},
			{name: "x", body: []byte("file-as-dir")},
			{name: "x/child.txt", body: []byte("child")},
			{name: "keep.txt", body: []byte("kept")},
		}
		z := archives.Zip{}
		ctx, cancel := testContext(t)
		defer cancel()

		result, err := z.ExtractParallel(ctx, newTrackingSource(buildZip(t, specs)),
			archives.ExtractParallelOptions{MaxWorkers: 2, ConflictPolicy: archives.ZipConflictSkip},
			func(ctx context.Context, file archives.FileInfo) error { return nil })
		if err != nil {
			t.Fatalf("skip policy must not return an error, got %v", err)
		}
		assertEveryOrdinalReported(t, result, len(specs))

		if result.Reports[0].Outcome != archives.ParallelEntryCompleted {
			t.Errorf("first duplicate outcome = %v, want completed", result.Reports[0].Outcome)
		}
		for _, idx := range []int{1, 3} {
			report := result.Reports[idx]
			if report.Outcome != archives.ParallelEntrySkipped || !errors.Is(report.Err, archives.ErrPathConflict) {
				t.Errorf("ordinal %d report = %+v, want conflict skip", idx, report)
			}
		}
		if result.Reports[4].Outcome != archives.ParallelEntryCompleted {
			t.Errorf("unrelated entry outcome = %v, want completed", result.Reports[4].Outcome)
		}
	})
}

func TestZipParallel_SymlinkTargetReadIsBudgetedAndNotFollowed(t *testing.T) {
	specs := []zipEntrySpec{
		{name: "link", symlinkTarget: "target-string"},
		{name: "regular.txt", body: []byte("ok")},
	}

	t.Run("target exposed and billed", func(t *testing.T) {
		z := archives.Zip{}
		ctx, cancel := testContext(t)
		defer cancel()

		var seenTarget string
		var seenSymlink bool
		result, err := z.ExtractParallel(ctx, newTrackingSource(buildZip(t, specs)),
			archives.ExtractParallelOptions{MaxWorkers: 2},
			func(ctx context.Context, file archives.FileInfo) error {
				if file.NameInArchive == "link" {
					seenSymlink = true
					seenTarget = file.LinkTarget
				}
				return nil
			})
		if err != nil {
			t.Fatalf("ExtractParallel failed: %v", err)
		}
		if !seenSymlink || seenTarget != "target-string" {
			t.Fatalf("symlink not presented correctly: isLink=%v target=%q", seenSymlink, seenTarget)
		}
		link := reportsByName(result)["link"]
		if link.Outcome != archives.ParallelEntryCompleted {
			t.Fatalf("link outcome = %v err = %v", link.Outcome, link.Err)
		}
		if link.BytesRead != int64(len("target-string")) {
			t.Errorf("link target billed %d bytes, want %d", link.BytesRead, len("target-string"))
		}
	})

	t.Run("tiny per-entry budget fails the link read", func(t *testing.T) {
		var handlerSawLink atomic.Bool
		z := archives.Zip{ContinueOnError: true}
		ctx, cancel := testContext(t)
		defer cancel()

		result, err := z.ExtractParallel(ctx, newTrackingSource(buildZip(t, specs)),
			archives.ExtractParallelOptions{MaxWorkers: 2, MaxEntryBytes: 4},
			func(ctx context.Context, file archives.FileInfo) error {
				if file.NameInArchive == "link" {
					handlerSawLink.Store(true)
				}
				return nil
			})
		if !errors.Is(err, archives.ErrEntrySizeLimit) {
			t.Fatalf("expected ErrEntrySizeLimit, got %v", err)
		}
		if handlerSawLink.Load() {
			t.Error("handler must not run when the symlink target cannot be read within budget")
		}
		link := reportsByName(result)["link"]
		if link.Outcome != archives.ParallelEntryFailed || !errors.Is(link.Err, archives.ErrEntrySizeLimit) {
			t.Fatalf("link report = %+v, want budget failure", link)
		}
	})
}

func TestZipParallel_ReaderLifecycle(t *testing.T) {
	const fileCount = 12
	specs := make([]zipEntrySpec, fileCount)
	for i := range specs {
		specs[i] = zipEntrySpec{
			name: fmt.Sprintf("f-%02d.txt", i),
			body: []byte(fmt.Sprintf("body-%02d", i)),
		}
	}

	t.Run("framework closes readers the handler leaves open", func(t *testing.T) {
		z := archives.Zip{}
		ctx, cancel := testContext(t)
		defer cancel()

		// MaxOpenReaders=1 would deadlock on any leaked reader slot.
		result, err := z.ExtractParallel(ctx, newTrackingSource(buildZip(t, specs)),
			archives.ExtractParallelOptions{MaxWorkers: 3, MaxOpenReaders: 1},
			func(ctx context.Context, file archives.FileInfo) error {
				handle, openErr := file.Open()
				if openErr != nil {
					return openErr
				}
				// Intentionally do not close before returning.
				if _, err := io.ReadAll(handle); err != nil {
					return err
				}
				return nil
			})
		if err != nil {
			t.Fatalf("ExtractParallel failed: %v", err)
		}
		assertEveryOrdinalReported(t, result, fileCount)
		for _, report := range result.Reports {
			if report.Outcome != archives.ParallelEntryCompleted {
				t.Errorf("entry %q outcome = %v err = %v", report.NameInArchive, report.Outcome, report.Err)
			}
		}
	})

	t.Run("handler-managed close and repeated open are safe", func(t *testing.T) {
		z := archives.Zip{}
		ctx, cancel := testContext(t)
		defer cancel()

		result, err := z.ExtractParallel(ctx, newTrackingSource(buildZip(t, specs)),
			archives.ExtractParallelOptions{MaxWorkers: 2},
			func(ctx context.Context, file archives.FileInfo) error {
				first, err := file.Open()
				if err != nil {
					return err
				}
				if _, err := io.Copy(io.Discard, first); err != nil {
					return err
				}
				if err := first.Close(); err != nil {
					return err
				}
				if err := first.Close(); err != nil { // idempotent
					return fmt.Errorf("double close failed: %w", err)
				}
				second, err := file.Open()
				if err != nil {
					return err
				}
				if _, err := io.Copy(io.Discard, second); err != nil {
					return err
				}
				return second.Close()
			})
		if err != nil {
			t.Fatalf("ExtractParallel failed: %v", err)
		}
		for _, report := range result.Reports {
			if report.Err != nil {
				t.Errorf("entry %q unexpected err: %v", report.NameInArchive, report.Err)
			}
		}
	})
}

func TestZipParallel_ContinueOnErrorStopVsContinue(t *testing.T) {
	const fileCount = 6
	specs := make([]zipEntrySpec, fileCount)
	for i := range specs {
		specs[i] = zipEntrySpec{name: fmt.Sprintf("f-%02d.txt", i), body: []byte("x")}
	}
	handler := func(ctx context.Context, file archives.FileInfo) error {
		if file.NameInArchive == "f-02.txt" || file.NameInArchive == "f-04.txt" {
			return errors.New("synthetic handler failure")
		}
		return nil
	}

	t.Run("continue aggregates every failure", func(t *testing.T) {
		z := archives.Zip{ContinueOnError: true}
		ctx, cancel := testContext(t)
		defer cancel()

		result, err := z.ExtractParallel(ctx, newTrackingSource(buildZip(t, specs)),
			archives.ExtractParallelOptions{MaxWorkers: 2}, handler)
		if err == nil {
			t.Fatal("expected aggregated handler errors")
		}
		assertEveryOrdinalReported(t, result, fileCount)
		var failed int
		for _, report := range result.Reports {
			if report.Outcome == archives.ParallelEntryFailed {
				failed++
			}
		}
		if failed != 2 {
			t.Errorf("failed count = %d, want 2", failed)
		}
	})

	t.Run("stop leaves later ordinals as not-run", func(t *testing.T) {
		z := archives.Zip{}
		ctx, cancel := testContext(t)
		defer cancel()

		result, err := z.ExtractParallel(ctx, newTrackingSource(buildZip(t, specs)),
			archives.ExtractParallelOptions{MaxWorkers: 1}, handler)
		if err == nil {
			t.Fatal("expected the handler error to stop the walk")
		}
		assertEveryOrdinalReported(t, result, fileCount)
		if reportsByName(result)["f-02.txt"].Outcome != archives.ParallelEntryFailed {
			t.Error("f-02.txt must be reported failed")
		}
		for i := 3; i < fileCount; i++ {
			name := fmt.Sprintf("f-%02d.txt", i)
			if reportsByName(result)[name].Outcome != archives.ParallelEntryNotRun {
				t.Errorf("%s outcome = %v, want not-run", name, reportsByName(result)[name].Outcome)
			}
		}
	})
}

func TestZipParallel_HandlerPanicIsRecoveredAndStopsTheWalk(t *testing.T) {
	const fileCount = 4
	specs := make([]zipEntrySpec, fileCount)
	for i := range specs {
		specs[i] = zipEntrySpec{name: fmt.Sprintf("f-%02d.txt", i), body: []byte("x")}
	}
	z := archives.Zip{}
	ctx, cancel := testContext(t)
	defer cancel()

	result, err := z.ExtractParallel(ctx, newTrackingSource(buildZip(t, specs)),
		archives.ExtractParallelOptions{MaxWorkers: 1},
		func(ctx context.Context, file archives.FileInfo) error {
			if file.NameInArchive == "f-01.txt" {
				panic("boom")
			}
			return nil
		})
	if err == nil || !strings.Contains(err.Error(), "file handler panicked") {
		t.Fatalf("expected the recovered panic in the returned error, got %v", err)
	}
	assertEveryOrdinalReported(t, result, fileCount)
	if reportsByName(result)["f-01.txt"].Outcome != archives.ParallelEntryFailed {
		t.Error("panicking entry must be reported failed")
	}
	// With a single worker, every later ordinal must be reported not-run.
	for i := 2; i < fileCount; i++ {
		name := fmt.Sprintf("f-%02d.txt", i)
		if reportsByName(result)[name].Outcome != archives.ParallelEntryNotRun {
			t.Errorf("%s outcome = %v, want not-run after panic", name,
				reportsByName(result)[name].Outcome)
		}
	}
}

func TestZipParallel_NonSeekableAndInvalidOptions(t *testing.T) {
	z := archives.Zip{}
	ctx, cancel := testContext(t)
	defer cancel()
	noop := func(context.Context, archives.FileInfo) error { return nil }

	if _, err := z.ExtractParallel(ctx, &bytes.Buffer{}, archives.ExtractParallelOptions{MaxWorkers: 1}, noop); err == nil {
		t.Error("expected error for non-seekable source")
	}
	for _, opts := range []archives.ExtractParallelOptions{
		{MaxWorkers: -1},
		{MaxOpenReaders: -2},
		{MaxDecompressedBytes: -1},
		{MaxEntryBytes: -1},
	} {
		if _, err := z.ExtractParallel(ctx, newTrackingSource(buildZip(t, []zipEntrySpec{{name: "a", body: []byte("b")}})),
			opts, noop); !errors.Is(err, archives.ErrInvalidParallelOptions) {
			t.Errorf("opts %+v: expected ErrInvalidParallelOptions, got %v", opts, err)
		}
	}
	if _, err := z.ExtractParallel(ctx, newTrackingSource(nil),
		archives.ExtractParallelOptions{MaxWorkers: 1}, nil); !errors.Is(err, archives.ErrInvalidParallelOptions) {
		t.Errorf("nil handler: expected ErrInvalidParallelOptions, got %v", err)
	}
}

func TestZipParallel_RaceSharedSource(t *testing.T) {
	const runs, fileCount, bodySize = 4, 20, 512
	specs := make([]zipEntrySpec, fileCount)
	for i := range specs {
		specs[i] = zipEntrySpec{
			name: fmt.Sprintf("f-%02d.bin", i),
			body: bytes.Repeat([]byte{byte(i)}, bodySize),
		}
	}
	source := newTrackingSource(buildZip(t, specs))
	source.delay = 2 * time.Millisecond
	z := archives.Zip{}

	var wg sync.WaitGroup
	for run := 0; run < runs; run++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()
			result, err := z.ExtractParallel(ctx, source,
				archives.ExtractParallelOptions{MaxWorkers: 3, MaxOpenReaders: 2},
				func(ctx context.Context, file archives.FileInfo) error {
					handle, err := file.Open()
					if err != nil {
						return err
					}
					defer handle.Close()
					data, err := io.ReadAll(handle)
					if err != nil {
						return err
					}
					if len(data) != bodySize {
						return fmt.Errorf("entry %q read %d bytes, want %d", file.NameInArchive, len(data), bodySize)
					}
					return nil
				})
			if err != nil {
				t.Errorf("concurrent run failed: %v", err)
				return
			}
			assertEveryOrdinalReported(t, result, fileCount)
			for _, report := range result.Reports {
				if report.Outcome != archives.ParallelEntryCompleted {
					t.Errorf("entry %q outcome %v err %v", report.NameInArchive, report.Outcome, report.Err)
				}
			}
		}()
	}
	wg.Wait()

	calls, peak, _, closes := source.snapshot()
	if calls == 0 {
		t.Error("no ReaderAt calls observed")
	}
	if peak < 2 {
		t.Errorf("shared source never saw concurrent ReadAt calls, peak = %d", peak)
	}
	if closes != 0 {
		t.Errorf("framework closed the shared source %d times", closes)
	}
	pos, err := source.Seek(0, io.SeekCurrent)
	if err != nil {
		t.Fatalf("seeking source position: %v", err)
	}
	if pos != 0 {
		t.Errorf("source seek position = %d, want 0 restored", pos)
	}
}

// ---------- corruption helpers ----------

// rewriteStored rebuilds one entry using the Store method so byte flips
// deterministically produce CRC errors rather than decompression failures.
func rewriteStored(t *testing.T, data []byte, name string) []byte {
	t.Helper()
	reader, err := stdzip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		t.Fatalf("opening zip: %v", err)
	}
	var specs []zipEntrySpec
	for _, file := range reader.File {
		rc, err := file.Open()
		if err != nil {
			t.Fatalf("opening %q: %v", file.FileHeader.Name, err)
		}
		body, err := io.ReadAll(rc)
		rc.Close()
		if err != nil {
			t.Fatalf("reading %q: %v", file.FileHeader.Name, err)
		}
		spec := zipEntrySpec{name: file.Name, body: body, dir: file.FileInfo().IsDir()}
		if file.Method == stdzip.Store && file.Name == name {
			// keep store for the target entry
		}
		specs = append(specs, spec)
	}
	var buf bytes.Buffer
	zw := stdzip.NewWriter(&buf)
	for _, spec := range specs {
		method := stdzip.Deflate
		if spec.dir || spec.name == name {
			method = stdzip.Store
		}
		hdr := &stdzip.FileHeader{Name: spec.name, Method: method}
		if spec.dir {
			hdr.Name += "/"
		}
		writer, err := zw.CreateHeader(hdr)
		if err != nil {
			t.Fatalf("create header: %v", err)
		}
		if !spec.dir {
			if _, err := writer.Write(spec.body); err != nil {
				t.Fatalf("write body: %v", err)
			}
		}
	}
	if err := zw.Close(); err != nil {
		t.Fatalf("close writer: %v", err)
	}
	return buf.Bytes()
}

func storedDataOffset(t *testing.T, data []byte, name string) int64 {
	t.Helper()
	reader, err := kzip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		t.Fatalf("opening zip with klauspost reader: %v", err)
	}
	for _, file := range reader.File {
		if file.Name == name {
			offset, err := file.DataOffset()
			if err != nil {
				t.Fatalf("data offset: %v", err)
			}
			return offset
		}
	}
	t.Fatalf("entry %q not found", name)
	return 0
}

func reportsByName(result archives.ExtractParallelResult) map[string]archives.ParallelEntryReport {
	byName := make(map[string]archives.ParallelEntryReport, len(result.Reports))
	for _, report := range result.Reports {
		byName[report.NameInArchive] = report
	}
	return byName
}
