package archives

import (
	"archive/tar"
	"archive/zip"
	"bytes"
	"compress/gzip"
	"context"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

// ---- test fixtures -------------------------------------------------------

type tarEntry struct {
	name string
	body []byte
	dir  bool
}

func makeTarBytes(t *testing.T, entries []tarEntry) []byte {
	t.Helper()
	var buf bytes.Buffer
	tw := tar.NewWriter(&buf)
	for _, e := range entries {
		if e.dir {
			if err := tw.WriteHeader(&tar.Header{
				Name:     e.name,
				Typeflag: tar.TypeDir,
				Mode:     0o755,
			}); err != nil {
				t.Fatalf("writing dir header %q: %v", e.name, err)
			}
			continue
		}
		hdr := &tar.Header{
			Name: e.name,
			Mode: 0o644,
			Size: int64(len(e.body)),
		}
		if err := tw.WriteHeader(hdr); err != nil {
			t.Fatalf("writing header %q: %v", e.name, err)
		}
		if _, err := tw.Write(e.body); err != nil {
			t.Fatalf("writing body %q: %v", e.name, err)
		}
	}
	if err := tw.Close(); err != nil {
		t.Fatalf("closing tar writer: %v", err)
	}
	return buf.Bytes()
}

func gzipBytes(t *testing.T, data []byte) []byte {
	t.Helper()
	var buf bytes.Buffer
	gw := gzip.NewWriter(&buf)
	if _, err := gw.Write(data); err != nil {
		t.Fatalf("gzip write: %v", err)
	}
	if err := gw.Close(); err != nil {
		t.Fatalf("gzip close: %v", err)
	}
	return buf.Bytes()
}

func makeZipBytes(t *testing.T, files map[string]string) []byte {
	t.Helper()
	var buf bytes.Buffer
	zw := zip.NewWriter(&buf)
	for name, content := range files {
		w, err := zw.Create(name)
		if err != nil {
			t.Fatalf("zip create %q: %v", name, err)
		}
		if _, err := w.Write([]byte(content)); err != nil {
			t.Fatalf("zip write %q: %v", name, err)
		}
	}
	if err := zw.Close(); err != nil {
		t.Fatalf("zip close: %v", err)
	}
	return buf.Bytes()
}

// putOctal writes value as a null-terminated octal field.
func putOctal(field []byte, value int64) {
	s := fmt.Sprintf("%0*o", len(field)-1, value)
	copy(field, []byte(s)[:len(field)-1])
	field[len(field)-1] = 0
}

// makeLyingTarBytes hand-crafts a tar whose single regular entry declares
// declaredSize in its header while only carrying a small real body. It
// models an adversary that inflates the declared size to exhaust naive,
// declaration-only resource limits.
func makeLyingTarBytes(t *testing.T, name string, declaredSize int64, realBody []byte) []byte {
	t.Helper()
	var out bytes.Buffer

	var hdr [512]byte
	copy(hdr[0:100], name)
	putOctal(hdr[100:108], 0o644)             // mode
	putOctal(hdr[108:116], 0)                 // uid
	putOctal(hdr[116:124], 0)                 // gid
	putOctal(hdr[124:136], declaredSize)      // size (the lie)
	putOctal(hdr[136:148], time.Now().Unix()) // mtime
	hdr[156] = tar.TypeReg                    // regular file
	// checksum field counts as spaces while summing
	for i := 148; i < 156; i++ {
		hdr[i] = ' '
	}
	var sum int
	for _, b := range hdr {
		sum += int(b)
	}
	copy(hdr[148:156], []byte(fmt.Sprintf("%06o\x00 ", sum)))
	out.Write(hdr[:])

	out.Write(realBody)
	if pad := len(realBody) % 512; pad != 0 {
		out.Write(make([]byte, 512-pad))
	}
	out.Write(make([]byte, 1024)) // two zero blocks = end of archive
	return out.Bytes()
}

func asBudgetError(t *testing.T, err error) *BudgetExceededError {
	t.Helper()
	var be *BudgetExceededError
	if !errors.As(err, &be) {
		t.Fatalf("expected BudgetExceededError, got %T: %v", err, err)
	}
	return be
}

// ---- entry counting ------------------------------------------------------

func TestBudgetManyEmptyEntries(t *testing.T) {
	entries := make([]tarEntry, 25)
	for i := range entries {
		entries[i] = tarEntry{name: fmt.Sprintf("empty-%02d", i), body: nil}
	}
	archive := makeTarBytes(t, entries)

	t.Run("limit enforced on empty entries", func(t *testing.T) {
		b := &ArchiveBudget{MaxEntries: 10}
		ctx := ContextWithBudget(context.Background(), b)
		err := Tar{}.Extract(ctx, bytes.NewReader(archive), func(_ context.Context, f FileInfo) error {
			return nil
		})
		be := asBudgetError(t, err)
		if be.Limit != LimitEntries {
			t.Errorf("expected limit %q, got %q", LimitEntries, be.Limit)
		}
		if b.Usage().Entries != 11 {
			t.Errorf("expected 11 entries metered at failure, got %d", b.Usage().Entries)
		}
		if !errors.Is(err, ErrBudgetExceeded) {
			t.Errorf("errors.Is(ErrBudgetExceeded) must hold")
		}
	})

	t.Run("no budget enumerates all entries", func(t *testing.T) {
		var seen int
		err := Tar{}.Extract(context.Background(), bytes.NewReader(archive), func(_ context.Context, f FileInfo) error {
			seen++
			return nil
		})
		if err != nil {
			t.Fatalf("extract: %v", err)
		}
		if seen != 25 {
			t.Errorf("expected 25 entries, got %d", seen)
		}
	})
}

func TestBudgetEntryLimitExactBoundary(t *testing.T) {
	archive := makeTarBytes(t, []tarEntry{
		{name: "a", body: []byte("a")},
		{name: "b", body: []byte("b")},
	})

	b := &ArchiveBudget{MaxEntries: 2} // exactly the number present
	ctx := ContextWithBudget(context.Background(), b)
	if err := (Tar{}).Extract(ctx, bytes.NewReader(archive), func(_ context.Context, _ FileInfo) error {
		return nil
	}); err != nil {
		t.Fatalf("limit equal to entry count must pass, got %v", err)
	}
}

// ---- fake declared size --------------------------------------------------

func TestBudgetFakeDeclaredSizeRejectedBeforeRead(t *testing.T) {
	lying := makeLyingTarBytes(t, "big.bin", 1<<30, []byte("tiny"))

	t.Run("per-entry limit", func(t *testing.T) {
		b := &ArchiveBudget{MaxEntryBytes: 1024}
		ctx := ContextWithBudget(context.Background(), b)
		var opened bool
		err := Tar{}.Extract(ctx, bytes.NewReader(lying), func(_ context.Context, f FileInfo) error {
			rc, err := f.Open()
			if err != nil {
				return err
			}
			defer rc.Close()
			opened = true
			_, err = io.ReadAll(rc)
			return err
		})
		be := asBudgetError(t, err)
		if be.Limit != LimitEntryBytes {
			t.Errorf("expected %q, got %q", LimitEntryBytes, be.Limit)
		}
		if be.Item != "big.bin" {
			t.Errorf("expected item big.bin, got %q", be.Item)
		}
		if opened {
			t.Errorf("handler must not receive a readable entry when declared size exceeds budget")
		}
		if b.Usage().DecompressedBytes != 0 {
			t.Errorf("no output bytes may be metered when the reader is rejected upfront, got %d", b.Usage().DecompressedBytes)
		}
	})

	t.Run("session total limit", func(t *testing.T) {
		b := &ArchiveBudget{MaxDecompressedBytes: 100}
		ctx := ContextWithBudget(context.Background(), b)
		err := Tar{}.Extract(ctx, bytes.NewReader(lying), func(_ context.Context, f FileInfo) error {
			rc, err := f.Open()
			if err != nil {
				return err
			}
			defer rc.Close()
			_, err = io.Copy(io.Discard, rc)
			return err
		})
		be := asBudgetError(t, err)
		if be.Limit != LimitDecompressedBytes {
			t.Errorf("expected %q, got %q", LimitDecompressedBytes, be.Limit)
		}
	})
}

// ---- actual byte accumulation -------------------------------------------

func TestBudgetActualBytesAccumulatedOnEveryRead(t *testing.T) {
	// unknown declared size: upfront checks are skipped and the limit can
	// only be enforced by accumulating actual emitted bytes
	b := &ArchiveBudget{MaxEntryBytes: 50, MaxDecompressedBytes: 500}
	if err := b.beforeEntry("mystery", 0, false); err != nil {
		t.Fatalf("unknown size must not fail upfront: %v", err)
	}
	rc := b.newEntryReader(io.NopCloser(bytes.NewReader(make([]byte, 1000))), "mystery")
	delivered := 0
	buf := make([]byte, 16)
	for {
		n, err := rc.Read(buf)
		delivered += n
		if errors.Is(err, ErrBudgetExceeded) {
			be := asBudgetError(t, err)
			if be.Limit != LimitEntryBytes {
				t.Errorf("expected %q, got %q", LimitEntryBytes, be.Limit)
			}
			break
		}
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	}
	if delivered <= 50 {
		t.Errorf("expected to cross the 50-byte limit, delivered %d", delivered)
	}
	if delivered > 50+len(buf) {
		t.Errorf("limit breach exceeded one read window: %d", delivered)
	}

	// equality semantics: exactly MaxDecompressedBytes is still allowed
	b2 := &ArchiveBudget{MaxDecompressedBytes: 100}
	rc2 := b2.newEntryReader(io.NopCloser(bytes.NewReader(make([]byte, 100))), "exact")
	data, err := io.ReadAll(rc2)
	if err != nil {
		t.Fatalf("reading exactly up to the limit must succeed: %v", err)
	}
	if len(data) != 100 {
		t.Errorf("expected 100 bytes, got %d", len(data))
	}
}

func TestBudgetRealCodecOutputMetered(t *testing.T) {
	// a highly compressible stream: small physical input, large output;
	// the gzip decoder's output must be metered on every Read even though
	// the declared tar size is bypassed by opening the codec directly
	raw := make([]byte, 20000)
	gz := gzipBytes(t, raw)

	b := &ArchiveBudget{MaxDecompressedBytes: 1024}
	dec, err := Gz{}.OpenReader(bytes.NewReader(gz))
	if err != nil {
		t.Fatalf("gzip reader: %v", err)
	}
	metered := b.newEntryReader(dec, "bomb.gz")
	_, err = io.Copy(io.Discard, metered)
	be := asBudgetError(t, err)
	if be.Limit != LimitDecompressedBytes {
		t.Errorf("expected %q, got %q", LimitDecompressedBytes, be.Limit)
	}
	// the breach is detected on the read that crosses the limit; io.Copy
	// uses a 32KiB buffer, so overshoot is bounded by one such read
	if used := b.Usage().DecompressedBytes; used <= 1024 || used > 1024+32768 {
		t.Errorf("output meter stopped just past the limit, got %d", used)
	}
}

// ---- Identify: physical vs CPU dimensions --------------------------------

func TestBudgetIdentifyPhysicalNotDoubleCounted(t *testing.T) {
	content := bytes.Repeat([]byte("this is plain text content "), 8) // ~200 bytes, no seeker

	t.Run("buffered rewind does not re-charge physical bytes", func(t *testing.T) {
		b := &ArchiveBudget{MaxIdentifyBytes: int64(len(content))}
		ctx := ContextWithBudget(context.Background(), b)
		_, _, err := Identify(ctx, "", bytes.NewBuffer(content))
		if !errors.Is(err, NoMatch) {
			t.Fatalf("expected NoMatch, got %v", err)
		}
		if used := b.Usage().IdentifyPhysicalBytes; used > int64(len(content)) {
			t.Errorf("physical meter charged %d bytes for a %d-byte buffered input; repeated probes must not double-count", used, len(content))
		}
	})

	t.Run("zero physical budget hard-fails instead of NoMatch", func(t *testing.T) {
		b := &ArchiveBudget{MaxIdentifyBytes: 1}
		ctx := ContextWithBudget(context.Background(), b)
		_, _, err := Identify(ctx, "", bytes.NewBuffer(content))
		if errors.Is(err, NoMatch) {
			t.Fatalf("budget failure must not be reported as NoMatch")
		}
		be := asBudgetError(t, err)
		if be.Limit != LimitIdentifyPhysicalBytes {
			t.Errorf("expected %q, got %q", LimitIdentifyPhysicalBytes, be.Limit)
		}
	})

	t.Run("seeker re-reads are physical reads and charged honestly", func(t *testing.T) {
		b := &ArchiveBudget{}
		ctx := ContextWithBudget(context.Background(), b)
		_, _, err := Identify(ctx, "", bytes.NewReader(content))
		if !errors.Is(err, NoMatch) {
			t.Fatalf("expected NoMatch, got %v", err)
		}
		if used := b.Usage().IdentifyPhysicalBytes; used < int64(len(content)) {
			t.Errorf("seeker re-reads should charge at least one full pass, got %d for %d-byte input", used, len(content))
		}
	})
}

func TestBudgetIdentifyCPURepeatedProbes(t *testing.T) {
	tarGz := gzipBytes(t, makeTarBytes(t, []tarEntry{{name: "f", body: []byte("hello")}}))

	t.Run("CPU limit aborts identification with a budget error", func(t *testing.T) {
		b := &ArchiveBudget{MaxIdentifyCPUBytes: 1}
		ctx := ContextWithBudget(context.Background(), b)
		_, _, err := Identify(ctx, "x.tar.gz", bytes.NewBuffer(tarGz))
		if errors.Is(err, NoMatch) {
			t.Fatalf("CPU budget failure must not become NoMatch")
		}
		be := asBudgetError(t, err)
		if be.Limit != LimitIdentifyCPUBytes {
			t.Errorf("expected %q, got %q", LimitIdentifyCPUBytes, be.Limit)
		}
	})

	t.Run("CPU and physical are separate meters", func(t *testing.T) {
		b := &ArchiveBudget{}
		ctx := ContextWithBudget(context.Background(), b)
		format, _, err := Identify(ctx, "x.tar.gz", bytes.NewBuffer(tarGz))
		if err != nil {
			t.Fatalf("identify: %v", err)
		}
		if format.Extension() != ".tar.gz" {
			t.Fatalf("expected .tar.gz, got %s", format.Extension())
		}
		usage := b.Usage()
		if usage.IdentifyCPUBytes == 0 {
			t.Errorf("repeated decompression probes must charge the CPU meter")
		}
		if usage.IdentifyPhysicalBytes > int64(len(tarGz)) {
			t.Errorf("physical meter charged %d for a %d-byte buffered input", usage.IdentifyPhysicalBytes, len(tarGz))
		}
	})
}

// ---- deep nesting, shared session ----------------------------------------

func buildNestedTarGz(t *testing.T, depth int) []byte {
	t.Helper()
	var inner []byte
	if depth > 1 {
		inner = buildNestedTarGz(t, depth-1)
	}
	entries := []tarEntry{{name: fmt.Sprintf("level%d.txt", depth), body: []byte("small-file")}}
	if depth > 1 {
		entries = append(entries, tarEntry{name: "inner.tar.gz", body: inner})
	} else {
		entries = append(entries, tarEntry{name: "leaf.txt", body: []byte("deep-leaf-content")})
	}
	return gzipBytes(t, makeTarBytes(t, entries))
}

type nestedCounters struct {
	bytes   int64
	entries int
}

func walkNested(ctx context.Context, t *testing.T, b *ArchiveBudget, data []byte, level int, seen *nestedCounters) error {
	fsys, err := FileSystem(ctx, fmt.Sprintf("level%d.tar.gz", level), bytes.NewReader(data))
	if err != nil {
		return err
	}
	afs, ok := fsys.(*ArchiveFS)
	if !ok {
		return fmt.Errorf("expected *ArchiveFS, got %T", fsys)
	}
	if afs.BudgetLevel() != level {
		t.Errorf("expected view level %d, got %d", level, afs.BudgetLevel())
	}
	entriesBefore := b.Usage().Entries
	dirEntries, err := fs.ReadDir(afs, ".")
	if err != nil {
		return err
	}
	if got := b.Usage().Entries - entriesBefore; got != int64(len(dirEntries)) {
		t.Errorf("first ReadDir metered %d entries, want %d", got, len(dirEntries))
	}
	// a second ReadDir must hit the cache and not re-enumerate entries
	cachedBefore := b.Usage().Entries
	if _, err := fs.ReadDir(afs, "."); err != nil {
		return err
	}
	if b.Usage().Entries != cachedBefore {
		t.Errorf("cached ReadDir created new entry charges: %d -> %d", cachedBefore, b.Usage().Entries)
	}
	for _, e := range dirEntries {
		f, err := afs.Open(e.Name())
		if err != nil {
			return err
		}
		body, err := io.ReadAll(f)
		if cerr := f.Close(); cerr != nil {
			t.Errorf("close: %v", cerr)
		}
		if err != nil {
			return err
		}
		seen.bytes += int64(len(body))
		seen.entries++
		if strings.HasSuffix(e.Name(), ".tar.gz") {
			childCtx := ContextWithBudgetLayer(ctx, b, afs.BudgetLevel(), afs.BudgetChain())
			if err := walkNested(childCtx, t, b, body, level+1, seen); err != nil {
				return err
			}
		}
	}
	return nil
}

func TestBudgetDeepNestedSmallPacks(t *testing.T) {
	const depth = 5
	pack := buildNestedTarGz(t, depth)

	t.Run("depth limit stops the descent with chain", func(t *testing.T) {
		b := &ArchiveBudget{MaxDepth: 3}
		ctx := ContextWithBudget(context.Background(), b)
		err := walkNested(ctx, t, b, pack, 1, &nestedCounters{})
		be := asBudgetError(t, err)
		if be.Limit != LimitDepth {
			t.Errorf("expected %q, got %q", LimitDepth, be.Limit)
		}
		if len(be.Chain) != 4 {
			t.Errorf("expected chain of 4 archives, got %v", be.Chain)
		}
		if b.Usage().Depth != 3 {
			t.Errorf("high-water depth should be 3, got %d", b.Usage().Depth)
		}
	})

	t.Run("meters are shared across layers and never reset", func(t *testing.T) {
		b := &ArchiveBudget{}
		ctx := ContextWithBudget(context.Background(), b)
		seen := &nestedCounters{}
		if err := walkNested(ctx, t, b, pack, 1, seen); err != nil {
			t.Fatalf("walk: %v", err)
		}
		usage := b.Usage()
		if usage.Depth != depth {
			t.Errorf("expected depth %d, got %d", depth, usage.Depth)
		}
		// Billing per layer (5 layers, 2 entries each):
		//   ReadDir indexes 2 headers; Open on the first entry re-scans 1
		//   header, Open on the second re-scans 2. Re-scans really parse
		//   headers again, so they are billed: 5 * (2+1+2) = 25.
		// What must NOT happen is a per-layer reset or a cached-ReadDir
		// surcharge, both of which are asserted inside walkNested.
		if usage.Entries != 25 {
			t.Errorf("expected 25 shared entry charges, got %d", usage.Entries)
		}
		if usage.DecompressedBytes != seen.bytes {
			t.Errorf("session output meter %d differs from bytes actually handed to handlers %d", usage.DecompressedBytes, seen.bytes)
		}

		// opening another root view on the same budget must observe the
		// accumulated state rather than a fresh allowance
		before := b.Usage()
		if _, err := FileSystem(ctx, "again.tar.gz", bytes.NewReader(pack)); err != nil {
			t.Fatalf("second root view: %v", err)
		}
		after := b.Usage()
		if after.Entries != before.Entries || after.DecompressedBytes != before.DecompressedBytes {
			t.Errorf("opening a view reset session meters: before=%+v after=%+v", before, after)
		}
		if after.Depth != before.Depth {
			t.Errorf("depth high-water mark changed unexpectedly: %d -> %d", before.Depth, after.Depth)
		}
	})
}

// ---- DeepFS cached inner views -------------------------------------------

func TestBudgetDeepFSCachedInnerView(t *testing.T) {
	dir := t.TempDir()
	zipPath := filepath.Join(dir, "arc.zip")
	zipBytes := makeZipBytes(t, map[string]string{"hello.txt": "hello world contents"})
	if err := os.WriteFile(zipPath, zipBytes, 0o644); err != nil {
		t.Fatalf("write zip: %v", err)
	}

	b := &ArchiveBudget{}
	fsys := &DeepFS{Root: dir, Budget: b}

	// first access: cache miss, identification is billed
	if _, err := fsys.ReadDir("arc.zip"); err != nil {
		t.Fatalf("first readdir: %v", err)
	}
	first := b.Usage()

	// second access: cached inner view + cached index, no new identification
	if _, err := fsys.ReadDir("arc.zip"); err != nil {
		t.Fatalf("second readdir: %v", err)
	}
	second := b.Usage()
	if second.IdentifyPhysicalBytes != first.IdentifyPhysicalBytes {
		t.Errorf("cache hit re-charged identification: %d -> %d", first.IdentifyPhysicalBytes, second.IdentifyPhysicalBytes)
	}
	if second.Entries != first.Entries {
		t.Errorf("cache hit re-enumerated entries: %d -> %d", first.Entries, second.Entries)
	}

	// re-opening and re-reading a file really decompresses it again, so
	// each read is billed against the output meter
	for i := 0; i < 2; i++ {
		f, err := fsys.Open("arc.zip/hello.txt")
		if err != nil {
			t.Fatalf("open %d: %v", i, err)
		}
		data, err := io.ReadAll(f)
		if err != nil {
			t.Fatalf("read %d: %v", i, err)
		}
		if err := f.Close(); err != nil {
			t.Fatalf("close %d: %v", i, err)
		}
		if string(data) != "hello world contents" {
			t.Fatalf("unexpected content %q", data)
		}
	}
	if got := b.Usage().DecompressedBytes; got != int64(2*len("hello world contents")) {
		t.Errorf("real re-decompression must be billed twice, got output=%d", got)
	}

	// Sub views share the exact budget pointer and level, no reset
	inner, err := fsys.getInnerFsys(zipPath)
	if err != nil {
		t.Fatalf("getInnerFsys: %v", err)
	}
	afs := inner.(*ArchiveFS)
	sub, err := afs.Sub(".")
	if err != nil {
		t.Fatalf("sub: %v", err)
	}
	subAFS := sub.(*ArchiveFS)
	if subAFS.Budget != b {
		t.Errorf("Sub view did not share the budget pointer")
	}
	if subAFS.BudgetLevel() != afs.BudgetLevel() {
		t.Errorf("Sub view changed the nesting level: %d -> %d", afs.BudgetLevel(), subAFS.BudgetLevel())
	}
}

// ---- ContinueOnError must not swallow budget errors ----------------------

func TestBudgetContinueOnErrorDoesNotSwallow(t *testing.T) {
	archive := makeZipBytes(t, map[string]string{
		"a.txt": "aaaaaaaaaa",
		"b.txt": "bbbbbbbbbb",
	})
	b := &ArchiveBudget{MaxEntryBytes: 1}
	ctx := ContextWithBudget(context.Background(), b)
	err := Zip{ContinueOnError: true}.Extract(ctx, bytes.NewReader(archive), func(_ context.Context, f FileInfo) error {
		rc, err := f.Open()
		if err != nil {
			return err
		}
		defer rc.Close()
		_, err = io.Copy(io.Discard, rc)
		return err
	})
	be := asBudgetError(t, err)
	if be.Limit != LimitEntryBytes {
		t.Errorf("expected %q, got %q", LimitEntryBytes, be.Limit)
	}
}

// ---- context cancellation stays independent ------------------------------

func TestBudgetContextCancellationIndependent(t *testing.T) {
	archive := makeTarBytes(t, []tarEntry{{name: "a", body: []byte("a")}})

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	err := Tar{}.Extract(ContextWithBudget(ctx, &ArchiveBudget{MaxEntries: 1}), bytes.NewReader(archive), func(_ context.Context, _ FileInfo) error {
		return nil
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", err)
	}
	if errors.Is(err, ErrBudgetExceeded) {
		t.Fatalf("cancellation must not be reported as a budget error")
	}
}

// ---- ownership: close once, never the caller's stream --------------------

type trackingReadCloser struct {
	io.Reader
	closes int
}

func (t *trackingReadCloser) Close() error {
	t.closes++
	return nil
}

type countCloser struct {
	closes int
}

func (c *countCloser) Read([]byte) (int, error) { return 0, io.EOF }
func (c *countCloser) Close() error             { c.closes++; return nil }

func TestBudgetReaderCloseOwnership(t *testing.T) {
	archive := makeTarBytes(t, []tarEntry{{name: "a", body: []byte("payload")}})

	callerStream := &trackingReadCloser{Reader: bytes.NewReader(archive)}
	b := &ArchiveBudget{}
	ctx := ContextWithBudget(context.Background(), b)
	err := Tar{}.Extract(ctx, callerStream, func(_ context.Context, f FileInfo) error {
		rc, err := f.Open()
		if err != nil {
			return err
		}
		if _, err := io.Copy(io.Discard, rc); err != nil {
			return err
		}
		if err := rc.Close(); err != nil {
			t.Errorf("first close: %v", err)
		}
		// repeated close must be harmless and close the lent reader once
		if err := rc.Close(); err != nil {
			t.Errorf("second close: %v", err)
		}
		return nil
	})
	if err != nil {
		t.Fatalf("extract: %v", err)
	}
	if callerStream.closes != 0 {
		t.Errorf("the caller's input stream must never be closed by the entry reader, closes=%d", callerStream.closes)
	}

	// the lent closer itself is closed exactly once
	lent := &countCloser{}
	rc := b.newEntryReader(lent, "a")
	_ = rc.Close()
	_ = rc.Close()
	if lent.closes != 1 {
		t.Errorf("lent reader closed %d times, want 1", lent.closes)
	}
}

// ---- restricted mode codec declaration -----------------------------------

func TestBudgetRestrictedCodecs(t *testing.T) {
	if got := SupportedBudgetCodecs(); len(got) == 0 || got[0] != ".br" {
		t.Errorf("supported codec list looks wrong: %v", got)
	}

	b := &ArchiveBudget{Restricted: true}
	if err := b.CheckRestricted(".gz"); err != nil {
		t.Errorf(".gz should be supported: %v", err)
	}
	err := b.CheckRestricted(".rar")
	if !errors.Is(err, ErrCodecNotBudgetable) {
		t.Errorf("rar must be refused in restricted mode, got %v", err)
	}
	err = b.CheckRestricted(".7z")
	if !errors.Is(err, ErrCodecNotBudgetable) {
		t.Errorf("7z must be refused in restricted mode, got %v", err)
	}

	// restricted RAR/7z extraction is refused before any decoder runs
	ctx := ContextWithBudget(context.Background(), b)
	if err := (Rar{}).Extract(ctx, bytes.NewReader(nil), nil); !errors.Is(err, ErrCodecNotBudgetable) {
		t.Errorf("rar extract: expected ErrCodecNotBudgetable, got %v", err)
	}
	if err := (SevenZip{}).Extract(ctx, bytes.NewReader(nil), nil); !errors.Is(err, ErrCodecNotBudgetable) {
		t.Errorf("7z extract: expected ErrCodecNotBudgetable, got %v", err)
	}

	// a supported codec keeps working under restriction
	tarGz := gzipBytes(t, makeTarBytes(t, []tarEntry{{name: "f", body: []byte("ok")}}))
	format, sr, err := Identify(ctx, "f.tar.gz", bytes.NewReader(tarGz))
	if err != nil {
		t.Fatalf("identify: %v", err)
	}
	var got int
	if err := format.(Extractor).Extract(ctx, sr, func(_ context.Context, f FileInfo) error {
		rc, err := f.Open()
		if err != nil {
			return err
		}
		defer rc.Close()
		body, err := io.ReadAll(rc)
		if err != nil {
			return err
		}
		got += len(body)
		return nil
	}); err != nil {
		t.Fatalf("restricted tar.gz extraction failed: %v", err)
	}
	if got != 2 {
		t.Errorf("expected 2 output bytes, got %d", got)
	}

	// without restriction, the same budget applies no codec gate
	open := &ArchiveBudget{}
	if err := open.CheckRestricted(".rar"); err != nil {
		t.Errorf("non-restricted budgets must not gate codecs: %v", err)
	}
	var nilBudget *ArchiveBudget
	if err := nilBudget.CheckRestricted(".rar"); err != nil {
		t.Errorf("nil budget must be a no-op: %v", err)
	}
}

// ---- concurrent access ----------------------------------------------------

func TestBudgetConcurrentCharging(t *testing.T) {
	b := &ArchiveBudget{} // no limits: every charge must be accounted exactly
	const goroutines, perG = 50, 100
	var wg sync.WaitGroup
	for i := 0; i < goroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			rc := b.newEntryReader(io.NopCloser(bytes.NewReader(make([]byte, perG))), "parallel")
			io.Copy(io.Discard, rc)
			_ = rc.Close()
		}()
	}
	wg.Wait()
	if got := b.Usage().DecompressedBytes; got != int64(goroutines*perG) {
		t.Errorf("concurrent metering lost charges: got %d, want %d", got, goroutines*perG)
	}
}

// ---- backward compatibility ----------------------------------------------

func TestBudgetNilIsInert(t *testing.T) {
	if b := BudgetFromContext(context.Background()); b != nil {
		t.Fatalf("unexpected budget in plain context")
	}
	var b *ArchiveBudget
	if err := b.countEntry("x"); err != nil {
		t.Errorf("nil countEntry: %v", err)
	}
	if err := b.beforeEntry("x", 1<<40, true); err != nil {
		t.Errorf("nil beforeEntry: %v", err)
	}
	if err := b.EnterLayer(99, []string{"x"}); err != nil {
		t.Errorf("nil EnterLayer: %v", err)
	}
	if u := b.Usage(); u != (BudgetUsage{}) {
		t.Errorf("nil usage should be zero value, got %+v", u)
	}
}
