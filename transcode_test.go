package archives

import (
	"archive/tar"
	"archive/zip"
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// This file verifies TranscodeArchive end to end. Output archives are always
// re-parsed with the standard library (archive/tar, archive/zip) rather than
// the package's own readers, so content and metadata checks are independent of
// the code under test.

// ---------- fixture builders / independent parsers ----------

type testEntry struct {
	name     string
	body     string
	dir      bool
	symlink  bool
	hardlink bool
	link     string
	mode     int64
	modTime  time.Time
	fifo     bool
	uid      int
	gid      int
}

func buildTarBytes(t *testing.T, entries []testEntry) []byte {
	t.Helper()
	var buf bytes.Buffer
	tw := tar.NewWriter(&buf)
	for _, e := range entries {
		hdr := &tar.Header{
			Name:    e.name,
			ModTime: e.modTime,
			Mode:    e.mode,
			Uid:     e.uid,
			Gid:     e.gid,
		}
		if hdr.Mode == 0 {
			hdr.Mode = 0o644
		}
		if hdr.ModTime.IsZero() {
			hdr.ModTime = time.Unix(1_700_000_000, 0)
		}
		switch {
		case e.dir:
			hdr.Typeflag = tar.TypeDir
			hdr.Mode = 0o755
		case e.symlink:
			hdr.Typeflag = tar.TypeSymlink
			hdr.Linkname = e.link
		case e.hardlink:
			hdr.Typeflag = tar.TypeLink
			hdr.Linkname = e.link
		case e.fifo:
			hdr.Typeflag = tar.TypeFifo
			hdr.Mode = int64(fs.ModeNamedPipe) | 0o644
		default:
			hdr.Typeflag = tar.TypeReg
			hdr.Size = int64(len(e.body))
		}
		if err := tw.WriteHeader(hdr); err != nil {
			t.Fatalf("write tar header %q: %v", e.name, err)
		}
		if hdr.Typeflag == tar.TypeReg {
			if _, err := tw.Write([]byte(e.body)); err != nil {
				t.Fatalf("write tar body %q: %v", e.name, err)
			}
		}
	}
	if err := tw.Close(); err != nil {
		t.Fatalf("close tar: %v", err)
	}
	return buf.Bytes()
}

func buildZipBytes(t *testing.T, entries []testEntry) []byte {
	t.Helper()
	var buf bytes.Buffer
	zw := zip.NewWriter(&buf)
	for _, e := range entries {
		hdr := &zip.FileHeader{Name: e.name, Method: zip.Deflate}
		if e.modTime.IsZero() {
			hdr.Modified = time.Unix(1_700_000_000, 0)
		} else {
			hdr.Modified = e.modTime
		}
		mode := fs.FileMode(e.mode)
		if mode == 0 {
			mode = 0o644
		}
		switch {
		case e.dir:
			hdr.SetMode(fs.ModeDir | 0o755)
			if !strings.HasSuffix(hdr.Name, "/") {
				hdr.Name += "/"
			}
		case e.symlink:
			hdr.SetMode(fs.ModeSymlink | 0o777)
		default:
			hdr.SetMode(mode)
		}
		w, err := zw.CreateHeader(hdr)
		if err != nil {
			t.Fatalf("zip create header %q: %v", e.name, err)
		}
		if e.symlink {
			if _, err := w.Write([]byte(e.link)); err != nil {
				t.Fatalf("zip write symlink %q: %v", e.name, err)
			}
			continue
		}
		if !e.dir {
			if _, err := w.Write([]byte(e.body)); err != nil {
				t.Fatalf("zip write body %q: %v", e.name, err)
			}
		}
	}
	if err := zw.Close(); err != nil {
		t.Fatalf("close zip: %v", err)
	}
	return buf.Bytes()
}

type parsedEntry struct {
	mode     fs.FileMode
	modTime  time.Time
	body     []byte
	link     string
	typeflag byte
}

func parseTar(t *testing.T, data []byte) map[string]parsedEntry {
	t.Helper()
	tr := tar.NewReader(bytes.NewReader(data))
	out := map[string]parsedEntry{}
	for {
		hdr, err := tr.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatalf("parse tar: %v", err)
		}
		pe := parsedEntry{mode: hdr.FileInfo().Mode(), modTime: hdr.ModTime, link: hdr.Linkname, typeflag: hdr.Typeflag}
		if hdr.Typeflag == tar.TypeReg {
			pe.body, err = io.ReadAll(tr)
			if err != nil {
				t.Fatalf("parse tar body %q: %v", hdr.Name, err)
			}
		}
		out[hdr.Name] = pe
	}
	return out
}

func parseZip(t *testing.T, data []byte) map[string]parsedEntry {
	t.Helper()
	zr, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		t.Fatalf("parse zip: %v", err)
	}
	out := map[string]parsedEntry{}
	for _, f := range zr.File {
		pe := parsedEntry{mode: f.Mode(), modTime: f.Modified}
		rc, err := f.Open()
		if err != nil {
			t.Fatalf("open zip entry %q: %v", f.Name, err)
		}
		pe.body, err = io.ReadAll(rc)
		rc.Close()
		if err != nil {
			t.Fatalf("read zip entry %q: %v", f.Name, err)
		}
		if f.Mode()&fs.ModeSymlink != 0 {
			pe.link = string(pe.body)
		}
		out[f.Name] = pe
	}
	return out
}

func transcode(t *testing.T, ta TranscodeArchive, input []byte) (*TranscodeResult, []byte) {
	t.Helper()
	var out bytes.Buffer
	res, err := ta.Transcode(context.Background(), bytes.NewReader(input), &out)
	if err != nil {
		t.Fatalf("transcode failed: %v", err)
	}
	return res, out.Bytes()
}

func assertGoroutinesSettled(t *testing.T, baseline int) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if runtime.NumGoroutine() <= baseline {
			return
		}
		time.Sleep(15 * time.Millisecond)
	}
	t.Fatalf("goroutine leak: now=%d baseline=%d", runtime.NumGoroutine(), baseline)
}

// ---------- fake Extractor / streams ----------

type fakeFileInfo struct {
	name    string
	size    int64
	mode    fs.FileMode
	modTime time.Time
}

func (f fakeFileInfo) Name() string       { return f.name }
func (f fakeFileInfo) Size() int64        { return f.size }
func (f fakeFileInfo) Mode() fs.FileMode  { return f.mode }
func (f fakeFileInfo) ModTime() time.Time { return f.modTime }
func (f fakeFileInfo) IsDir() bool        { return f.mode&fs.ModeDir != 0 }
func (f fakeFileInfo) Sys() any           { return nil }

type fakeFile struct {
	io.Reader
	info fs.FileInfo
}

func (f fakeFile) Close() error               { return nil }
func (f fakeFile) Stat() (fs.FileInfo, error) { return f.info, nil }

// fakeExtractor hands out FileInfo.Open handles that are only valid while the
// handler is running, exactly like Tar's tar.Reader-backed entries.
type fakeExtractor struct {
	entries      []testEntry
	openErrors   map[string]error
	expireBorrow bool
}

func (f fakeExtractor) Extract(ctx context.Context, _ io.Reader, handle FileHandler) error {
	for _, e := range f.entries {
		if err := ctx.Err(); err != nil {
			return err
		}
		mode := fs.FileMode(e.mode)
		if mode == 0 {
			mode = 0o644
		}
		switch {
		case e.dir:
			mode |= fs.ModeDir
		case e.symlink:
			mode = fs.ModeSymlink | 0o777
		}
		info := fakeFileInfo{name: path.Base(e.name), size: int64(len(e.body)), mode: mode, modTime: time.Unix(1_700_000_000, 0)}
		valid := true
		fi := FileInfo{
			FileInfo:      info,
			NameInArchive: e.name,
			LinkTarget:    e.link,
			Open: func() (fs.File, error) {
				if f.expireBorrow && !valid {
					return nil, errors.New("fake extractor: entry borrow expired after handler return")
				}
				if err := f.openErrors[e.name]; err != nil {
					return nil, err
				}
				return fakeFile{Reader: strings.NewReader(e.body), info: info}, nil
			},
		}
		if err := handle(ctx, fi); err != nil {
			return err
		}
		valid = false
	}
	return nil
}

type gateWriter struct {
	gate    chan struct{}
	written atomic.Int64
}

func (g *gateWriter) Write(p []byte) (int, error) {
	<-g.gate
	g.written.Add(int64(len(p)))
	return len(p), nil
}

type failWriter struct {
	failAfter atomic.Int64
	calls     atomic.Int64
}

func (w *failWriter) Write(p []byte) (int, error) {
	w.calls.Add(1)
	return 0, errors.New("failWriter: simulated target write failure")
}

type closeTracker struct{ closed atomic.Bool }

type trackingReadCloser struct {
	closeTracker
	io.Reader
}

func (t *trackingReadCloser) Close() error {
	t.closed.Store(true)
	return nil
}

type trackingWriteCloser struct {
	closeTracker
	io.Writer
}

func (t *trackingWriteCloser) Close() error {
	t.closed.Store(true)
	return nil
}

// holdingTarget is an ArchiverAsync test double that records job ordering and
// in-flight concurrency. It never reads entry content.
type holdingTarget struct {
	gate     chan struct{}
	mu       sync.Mutex
	order    []string
	inFlight atomic.Int64
	maxSeen  atomic.Int64
}

func (h *holdingTarget) Archive(_ context.Context, _ io.Writer, _ []FileInfo) error {
	return errors.New("holdingTarget: Archive is not used in tests")
}

func (h *holdingTarget) ArchiveAsync(ctx context.Context, _ io.Writer, jobs <-chan ArchiveAsyncJob) error {
	// Receive eagerly and process in parallel so the feeder can build up
	// unacknowledged jobs up to the caller's concurrency budget. Reception
	// stays single-threaded so arrival order is preserved.
	var wg sync.WaitGroup
	for job := range jobs {
		cur := h.inFlight.Add(1)
		for {
			max := h.maxSeen.Load()
			if cur <= max || h.maxSeen.CompareAndSwap(max, cur) {
				break
			}
		}
		h.mu.Lock()
		h.order = append(h.order, job.File.NameInArchive)
		h.mu.Unlock()
		wg.Add(1)
		go func(job ArchiveAsyncJob) {
			defer wg.Done()
			select {
			case <-h.gate:
				job.Result <- nil
			case <-ctx.Done():
				job.Result <- ctx.Err()
			}
			h.inFlight.Add(-1)
		}(job)
	}
	wg.Wait()
	return nil
}

func countSpoolDirs(t *testing.T) int {
	t.Helper()
	dir, err := os.ReadDir(os.TempDir())
	if err != nil {
		t.Fatalf("read temp dir: %v", err)
	}
	n := 0
	for _, d := range dir {
		if strings.HasPrefix(d.Name(), "archives-transcode-") {
			n++
		}
	}
	return n
}

// ---------- tests ----------

func TestTranscode_TarToZip_WithDirectories(t *testing.T) {
	mod := time.Unix(1_700_123_456, 0)
	src := buildTarBytes(t, []testEntry{
		{name: "sub/", dir: true, modTime: mod},
		{name: "a.txt", body: "alpha", mode: 0o640, modTime: mod},
		{name: "sub/b.txt", body: "beta-beta", mode: 0o600, modTime: mod},
	})

	res, out := transcode(t, TranscodeArchive{Source: Tar{}, Target: Zip{}}, src)
	entries := parseZip(t, out)

	if got, want := res.EntriesSeen, int64(3); got != want {
		t.Errorf("EntriesSeen=%d want %d", got, want)
	}
	if got, want := res.EntriesWritten, int64(3); got != want {
		t.Errorf("EntriesWritten=%d want %d", got, want)
	}
	if res.Incomplete {
		t.Errorf("result should not be incomplete: %+v", res.Errors)
	}
	if len(entries) != 3 {
		t.Fatalf("expected 3 zip entries, got %d: %v", len(entries), mapKeys(entries))
	}
	dir, ok := entries["sub/"]
	if !ok || dir.mode&fs.ModeDir == 0 {
		t.Errorf("directory entry sub/ missing or not a dir: %+v", dir)
	}
	a := entries["a.txt"]
	if string(a.body) != "alpha" {
		t.Errorf("a.txt body=%q want alpha", a.body)
	}
	if a.mode.Perm() != 0o640 {
		t.Errorf("a.txt perms=%o want 640", a.mode.Perm())
	}
	if d := a.modTime.Sub(mod).Abs(); d > 2*time.Second {
		t.Errorf("a.txt modtime drifted by %v", d)
	}
	b := entries["sub/b.txt"]
	if string(b.body) != "beta-beta" || b.mode.Perm() != 0o600 {
		t.Errorf("sub/b.txt mismatch: body=%q perms=%o", b.body, b.mode.Perm())
	}
}

func TestTranscode_ZipToTar_WithDirectories(t *testing.T) {
	src := buildZipBytes(t, []testEntry{
		{name: "dir/", dir: true},
		{name: "dir/c.txt", body: "see"},
		{name: "root.txt", body: "root"},
	})

	res, out := transcode(t, TranscodeArchive{Source: Zip{}, Target: Tar{}}, src)
	entries := parseTar(t, out)

	if res.Incomplete {
		t.Errorf("unexpected incomplete result: %+v", res.Errors)
	}
	if len(entries) != 3 {
		t.Fatalf("expected 3 tar entries, got %d: %v", len(entries), mapKeys(entries))
	}
	// Normalization strips the trailing slash; the directory type bits survive.
	if entries["dir"].typeflag != tar.TypeDir || entries["dir"].mode&fs.ModeDir == 0 {
		t.Errorf("dir typeflag=%d mode=%v, want TypeDir", entries["dir"].typeflag, entries["dir"].mode)
	}
	if entries["dir/c.txt"].typeflag != tar.TypeReg || string(entries["dir/c.txt"].body) != "see" {
		t.Errorf("dir/c.txt mismatch: %+v", entries["dir/c.txt"])
	}
	if string(entries["root.txt"].body) != "root" {
		t.Errorf("root.txt body=%q", entries["root.txt"].body)
	}
}

func TestTranscode_FilterRename(t *testing.T) {
	src := buildTarBytes(t, []testEntry{
		{name: "keep.txt", body: "k"},
		{name: "skip.txt", body: "s"},
		{name: "old/name.txt", body: "n"},
	})
	ta := TranscodeArchive{
		Source: Tar{},
		Target: Zip{},
		Filter: func(info FileInfo) (string, bool) {
			if info.NameInArchive == "skip.txt" {
				return "", false
			}
			return "new/" + info.NameInArchive, true
		},
	}
	res, out := transcode(t, ta, src)
	entries := parseZip(t, out)

	if res.EntriesSkipped != 1 || res.EntriesWritten != 2 {
		t.Errorf("skipped=%d written=%d", res.EntriesSkipped, res.EntriesWritten)
	}
	if _, ok := entries["new/skip.txt"]; ok {
		t.Error("skipped entry present in output")
	}
	if string(entries["new/keep.txt"].body) != "k" {
		t.Error("keep.txt missing or wrong body after rename")
	}
	if string(entries["new/old/name.txt"].body) != "n" {
		t.Error("old/name.txt missing or wrong body after rename")
	}
}

func TestTranscode_SymlinkPolicies(t *testing.T) {
	src := buildTarBytes(t, []testEntry{
		{name: "target.txt", body: "payload"},
		{name: "link", symlink: true, link: "target.txt"},
		{name: "abslink", symlink: true, link: "/outside/x"},
	})

	// Preserve TAR -> ZIP: link semantics survive, body is the target string.
	_, out := transcode(t, TranscodeArchive{Source: Tar{}, Target: Zip{}, Symlinks: LinkPreserve}, src)
	zipEntries := parseZip(t, out)
	link := zipEntries["link"]
	if link.mode&fs.ModeSymlink == 0 {
		t.Fatalf("link should remain a symlink in zip, mode=%v", link.mode)
	}
	if link.link != "target.txt" {
		t.Errorf("zip symlink target=%q want target.txt", link.link)
	}
	if zipEntries["abslink"].link != "/outside/x" {
		t.Errorf("absolute symlink target altered: %q", zipEntries["abslink"].link)
	}

	// Preserve ZIP -> TAR: typeflag and linkname survive.
	res, out := transcode(t, TranscodeArchive{Source: Zip{}, Target: Tar{}, Symlinks: LinkPreserve}, out)
	tarEntries := parseTar(t, out)
	if tarEntries["link"].typeflag != tar.TypeSymlink || tarEntries["link"].link != "target.txt" {
		t.Errorf("tar symlink not preserved: %+v", tarEntries["link"])
	}
	if res.Incomplete {
		t.Errorf("unexpected incomplete: %+v", res.Errors)
	}

	// Reject.
	if _, err := (TranscodeArchive{Source: Tar{}, Target: Zip{}, Symlinks: LinkReject}).Transcode(
		context.Background(), bytes.NewReader(src), &bytes.Buffer{}); !errors.Is(err, ErrUnsupportedLink) {
		t.Errorf("expected ErrUnsupportedLink on reject, got %v", err)
	}

	// Materialize without the target retained: explicit error, never a disk read.
	if _, err := (TranscodeArchive{Source: Tar{}, Target: Zip{}, Symlinks: LinkMaterialize}).Transcode(
		context.Background(), bytes.NewReader(src), &bytes.Buffer{}); !errors.Is(err, ErrUnsupportedLink) {
		t.Errorf("expected ErrUnsupportedLink on unspoolable materialize, got %v", err)
	}

	// Materialize with the spool active: the resolvable symlink becomes a
	// regular file with the in-archive target's content; disk paths are never
	// followed. Use a fixture whose links all point inside the archive.
	materializable := buildTarBytes(t, []testEntry{
		{name: "target.txt", body: "payload"},
		{name: "link", symlink: true, link: "target.txt"},
	})
	_, out = transcode(t, TranscodeArchive{
		Source:    Tar{},
		Target:    Zip{},
		Symlinks:  LinkMaterialize,
		Hardlinks: LinkMaterialize, // activates the private spool
	}, materializable)
	zipEntries = parseZip(t, out)
	mat := zipEntries["link"]
	if mat.mode&fs.ModeSymlink != 0 {
		t.Errorf("materialized link should be regular, mode=%v", mat.mode)
	}
	if string(mat.body) != "payload" {
		t.Errorf("materialized body=%q want payload", mat.body)
	}
}

func TestTranscode_HardlinkPolicies(t *testing.T) {
	// Real-world fixture: file-a (regular) precedes file-b (hard link to it).
	f, err := os.Open("testdata/gnu-hardlinks.tar")
	if err != nil {
		t.Fatalf("open fixture: %v", err)
	}
	stats, _ := f.Stat()
	src := make([]byte, stats.Size())
	if _, err := io.ReadFull(f, src); err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	f.Close()

	// Materialize TAR -> ZIP: both entries are regular, same content, no empties.
	_, out := transcode(t, TranscodeArchive{Source: Tar{}, Target: Zip{}, Hardlinks: LinkMaterialize}, src)
	zipEntries := parseZip(t, out)
	a := zipEntries["dir-1/dir-2/file-a"]
	b := zipEntries["dir-1/dir-2/file-b"]
	if len(a.body) == 0 {
		t.Fatal("file-a unexpectedly empty")
	}
	if string(a.body) != string(b.body) {
		t.Errorf("materialized hard link body mismatch: a=%q b=%q", a.body, b.body)
	}
	if b.mode&fs.ModeSymlink != 0 {
		t.Errorf("materialized hard link must be regular, mode=%v", b.mode)
	}

	// Preserve into Zip is impossible and must say so explicitly.
	if _, err := (TranscodeArchive{Source: Tar{}, Target: Zip{}, Hardlinks: LinkPreserve}).Transcode(
		context.Background(), bytes.NewReader(src), &bytes.Buffer{}); !errors.Is(err, ErrUnsupportedLink) {
		t.Errorf("preserve hardlink into zip: expected ErrUnsupportedLink, got %v", err)
	}
	// Reject.
	if _, err := (TranscodeArchive{Source: Tar{}, Target: Zip{}, Hardlinks: LinkReject}).Transcode(
		context.Background(), bytes.NewReader(src), &bytes.Buffer{}); !errors.Is(err, ErrUnsupportedLink) {
		t.Errorf("reject hardlink: expected ErrUnsupportedLink, got %v", err)
	}

	// Preserve TAR -> TAR: TypeLink and Linkname survive.
	_, out = transcode(t, TranscodeArchive{Source: Tar{}, Target: Tar{}, Hardlinks: LinkPreserve}, src)
	tarEntries := parseTar(t, out)
	if tarEntries["dir-1/dir-2/file-b"].typeflag != tar.TypeLink {
		t.Errorf("expected TypeLink, got %d", tarEntries["dir-1/dir-2/file-b"].typeflag)
	}
	if tarEntries["dir-1/dir-2/file-b"].link != "dir-1/dir-2/file-a" {
		t.Errorf("linkname=%q", tarEntries["dir-1/dir-2/file-b"].link)
	}

	// Spool budget too small: explicit failure, never an empty file.
	_, err = (TranscodeArchive{Source: Tar{}, Target: Zip{}, Hardlinks: LinkMaterialize, MaxSpoolBytes: 1}).Transcode(
		context.Background(), bytes.NewReader(src), &bytes.Buffer{})
	if !errors.Is(err, ErrSpoolLimit) {
		t.Errorf("expected ErrSpoolLimit, got %v", err)
	}
}

func TestTranscode_HardlinkForwardReference(t *testing.T) {
	// The link appears BEFORE its regular target.
	src := buildTarBytes(t, []testEntry{
		{name: "b.txt", hardlink: true, link: "a.txt"},
		{name: "a.txt", body: "forward-payload"},
	})
	_, out := transcode(t, TranscodeArchive{Source: Tar{}, Target: Zip{}, Hardlinks: LinkMaterialize}, src)
	entries := parseZip(t, out)
	if string(entries["a.txt"].body) != "forward-payload" {
		t.Errorf("a.txt body=%q", entries["a.txt"].body)
	}
	if string(entries["b.txt"].body) != "forward-payload" {
		t.Errorf("forward-referenced hard link body=%q, must never be an empty file", entries["b.txt"].body)
	}
	if len(entries["b.txt"].body) == 0 {
		t.Fatal("silent empty placeholder written for forward reference")
	}

	// Dangling link: target never appears.
	dangling := buildTarBytes(t, []testEntry{
		{name: "b.txt", hardlink: true, link: "missing.txt"},
		{name: "a.txt", body: "x"},
	})
	if _, err := (TranscodeArchive{Source: Tar{}, Target: Zip{}, Hardlinks: LinkMaterialize}).Transcode(
		context.Background(), bytes.NewReader(dangling), &bytes.Buffer{}); !errors.Is(err, ErrDanglingLink) {
		t.Errorf("expected ErrDanglingLink, got %v", err)
	}
}

func TestTranscode_ConflictPolicies(t *testing.T) {
	src := buildTarBytes(t, []testEntry{
		{name: "one.txt", body: "1"},
		{name: "two.txt", body: "22"},
	})
	forceDup := func(info FileInfo) (string, bool) { return "dup.txt", true }

	// Default: hard error.
	if _, err := (TranscodeArchive{Source: Tar{}, Target: Zip{}, Filter: forceDup}).Transcode(
		context.Background(), bytes.NewReader(src), &bytes.Buffer{}); !errors.Is(err, ErrNameConflict) {
		t.Errorf("expected ErrNameConflict, got %v", err)
	}

	// Skip.
	res, out := transcode(t, TranscodeArchive{Source: Tar{}, Target: Zip{}, Filter: forceDup, OnConflict: ConflictSkip}, src)
	entries := parseZip(t, out)
	if len(entries) != 1 || string(entries["dup.txt"].body) != "1" {
		t.Errorf("skip policy mismatch: %+v", entries)
	}
	if res.EntriesSkipped != 1 {
		t.Errorf("EntriesSkipped=%d want 1", res.EntriesSkipped)
	}

	// Suffix.
	_, out = transcode(t, TranscodeArchive{Source: Tar{}, Target: Zip{}, Filter: forceDup, OnConflict: ConflictSuffix}, src)
	entries = parseZip(t, out)
	if string(entries["dup.txt"].body) != "1" || string(entries["dup-1.txt"].body) != "22" {
		t.Errorf("suffix policy mismatch: %v", mapKeys(entries))
	}
}

func TestTranscode_HardLimits(t *testing.T) {
	src := buildTarBytes(t, []testEntry{
		{name: "a", body: "0123456789"},
		{name: "b", body: "0123456789"},
		{name: "c", body: "0123456789"},
	})

	_, err := (TranscodeArchive{Source: Tar{}, Target: Zip{}, MaxEntries: 2}).Transcode(
		context.Background(), bytes.NewReader(src), &bytes.Buffer{})
	if !errors.Is(err, ErrMaxEntries) {
		t.Errorf("MaxEntries: expected ErrMaxEntries, got %v", err)
	}

	_, err = (TranscodeArchive{Source: Tar{}, Target: Zip{}, MaxDecompressedBytes: 15}).Transcode(
		context.Background(), bytes.NewReader(src), &bytes.Buffer{})
	if !errors.Is(err, ErrDecompressedLimit) {
		t.Errorf("MaxDecompressedBytes: expected ErrDecompressedLimit, got %v", err)
	}

	// Spool limit with materialization (declared size checked up front).
	_, err = (TranscodeArchive{Source: Tar{}, Target: Zip{}, Hardlinks: LinkMaterialize, MaxSpoolBytes: 5}).Transcode(
		context.Background(), bytes.NewReader(src), &bytes.Buffer{})
	if !errors.Is(err, ErrSpoolLimit) {
		t.Errorf("MaxSpoolBytes: expected ErrSpoolLimit, got %v", err)
	}
}

func TestTranscode_SlowWriterAndCancellation(t *testing.T) {
	baseline := runtime.NumGoroutine()
	src := buildTarBytes(t, []testEntry{
		{name: "a", body: strings.Repeat("x", 4096)},
		{name: "b", body: strings.Repeat("y", 4096)},
	})

	gate := make(chan struct{})
	w := &gateWriter{gate: gate}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		_, err := (TranscodeArchive{Source: Tar{}, Target: Tar{}}).Transcode(ctx, bytes.NewReader(src), w)
		done <- err
	}()

	// Give the target time to block inside the slow writer, then cancel.
	time.Sleep(100 * time.Millisecond)
	cancel()
	select {
	case <-done:
		t.Fatal("transcode must not return while the caller's writer is still blocked")
	case <-time.After(100 * time.Millisecond):
	}

	// Releasing the writer must let every goroutine finish; no handshake deadlock.
	close(gate)
	select {
	case err := <-done:
		if err == nil {
			t.Error("expected an error after cancellation")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("transcode deadlocked after writer was released")
	}
	assertGoroutinesSettled(t, baseline+1) // +1 for this test's own goroutine tree tolerance
}

func TestTranscode_TargetWriterFailure(t *testing.T) {
	baseline := runtime.NumGoroutine()
	src := buildTarBytes(t, []testEntry{{name: "a", body: "hello"}})

	start := time.Now()
	_, err := (TranscodeArchive{Source: Tar{}, Target: Zip{}}).Transcode(
		context.Background(), bytes.NewReader(src), &failWriter{})
	if err == nil {
		t.Fatal("target write/finalize error must be returned")
	}
	if time.Since(start) > 5*time.Second {
		t.Fatalf("handshake deadlocked on target failure for %v", time.Since(start))
	}
	assertGoroutinesSettled(t, baseline+1)
}

func TestTranscode_SourceReadError(t *testing.T) {
	baseline := runtime.NumGoroutine()
	full := buildTarBytes(t, []testEntry{
		{name: "a", body: strings.Repeat("a", 2048)},
		{name: "b", body: strings.Repeat("b", 2048)},
		{name: "c", body: strings.Repeat("c", 2048)},
	})
	corrupt := append([]byte(nil), full[:len(full)*2/3]...)

	start := time.Now()
	_, err := (TranscodeArchive{Source: Tar{}, Target: Tar{}}).Transcode(
		context.Background(), bytes.NewReader(corrupt), &bytes.Buffer{})
	if err == nil {
		t.Fatal("expected source read error to surface")
	}
	if time.Since(start) > 5*time.Second {
		t.Fatalf("deadlocked on source error for %v", time.Since(start))
	}
	assertGoroutinesSettled(t, baseline+1)
}

func TestTranscode_EntryBorrowScope(t *testing.T) {
	// The fake Extractor invalidates Open as soon as each handler returns.
	// Correct output proves every body is consumed within the handler window.
	ta := TranscodeArchive{
		Source: fakeExtractor{
			expireBorrow: true,
			entries: []testEntry{
				{name: "f1.txt", body: "borrow-one"},
				{name: "dir/f2.txt", body: "borrow-two"},
				{name: "f3.txt", body: "borrow-three"},
			},
		},
		Target: Tar{},
	}
	var out bytes.Buffer
	res, err := ta.run(context.Background(), nil, &out)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	entries := parseTar(t, out.Bytes())
	if res.EntriesWritten != 3 {
		t.Fatalf("written=%d want 3", res.EntriesWritten)
	}
	for name, want := range map[string]string{
		"f1.txt":     "borrow-one",
		"dir/f2.txt": "borrow-two",
		"f3.txt":     "borrow-three",
	} {
		if string(entries[name].body) != want {
			t.Errorf("%s body=%q want %q (entry read outside borrow window?)", name, entries[name].body, want)
		}
	}
}

func TestTranscode_DoesNotCloseCallerStreams(t *testing.T) {
	src := buildTarBytes(t, []testEntry{{name: "a", body: "data"}})
	in := &trackingReadCloser{Reader: bytes.NewReader(src)}
	out := &trackingWriteCloser{Writer: &bytes.Buffer{}}

	if _, err := (TranscodeArchive{Source: Tar{}, Target: Zip{}}).Transcode(
		context.Background(), in, out); err != nil {
		t.Fatalf("transcode: %v", err)
	}
	if in.closed.Load() {
		t.Error("input stream was closed without permission")
	}
	if out.closed.Load() {
		t.Error("output stream was closed without permission")
	}
}

func TestTranscode_ContinueOnError(t *testing.T) {
	// Failures happen before any target byte is written, so the finalized
	// archive stays valid and the partial output is explicitly marked.
	ta := TranscodeArchive{
		Source: fakeExtractor{
			entries: []testEntry{
				{name: "good1.txt", body: "one"},
				{name: "../escape.txt", body: "bad"}, // rejected by name normalization
				{name: "link", symlink: true, link: "good1.txt"},
				{name: "good2.txt", body: "three"},
			},
		},
		Target:          Tar{},
		Symlinks:        LinkReject, // the symlink entry becomes a tolerated error
		ContinueOnError: true,
	}
	var out bytes.Buffer
	res, err := ta.run(context.Background(), nil, &out)
	if err != nil {
		t.Fatalf("ContinueOnError should not return an error, got %v", err)
	}
	if !res.Incomplete || len(res.Errors) < 2 {
		t.Fatalf("partial output must be marked incomplete with recorded errors, res=%+v", res)
	}
	if res.EntriesWritten != 2 {
		t.Errorf("written=%d want 2", res.EntriesWritten)
	}
	// Target archive must still be fully finalized and independently readable.
	entries := parseTar(t, out.Bytes())
	if string(entries["good1.txt"].body) != "one" || string(entries["good2.txt"].body) != "three" {
		t.Errorf("tolerated entries mismatch: %v", mapKeys(entries))
	}
	for _, bad := range []string{"../escape.txt", "escape.txt", "link"} {
		if _, ok := entries[bad]; ok {
			t.Errorf("rejected entry %q should not be present", bad)
		}
	}

	// The same failures without ContinueOnError must abort hard.
	if _, err := ta.withContinue(false).run(context.Background(), nil, &bytes.Buffer{}); err == nil {
		t.Error("expected a hard error when ContinueOnError is disabled")
	}
}

func (ta TranscodeArchive) withContinue(v bool) TranscodeArchive {
	ta.ContinueOnError = v
	return ta
}

func TestTranscode_SpoolCleanup(t *testing.T) {
	before := countSpoolDirs(t)
	src := buildTarBytes(t, []testEntry{
		{name: "a.txt", body: "spool-me"},
		{name: "b.txt", hardlink: true, link: "a.txt"},
	})
	ta := TranscodeArchive{Source: Tar{}, Target: Zip{}, Hardlinks: LinkMaterialize}
	res, _ := transcode(t, ta, src)
	if res.SpoolBytesPeak == 0 {
		t.Error("expected spool usage under materialization")
	}
	if got := countSpoolDirs(t); got != before {
		t.Errorf("private spool not reclaimed: before=%d after=%d", before, got)
	}
}

func TestTranscode_MetadataPolicies(t *testing.T) {
	src := buildTarBytes(t, []testEntry{
		{name: "owned.txt", body: "x", uid: 1000, gid: 1000},
		{name: "pipe", fifo: true},
	})

	// Reject: Unix ownership has no Zip representation.
	_, err := (TranscodeArchive{Source: Tar{}, Target: Zip{}, Metadata: MetadataReject}).Transcode(
		context.Background(), bytes.NewReader(src), &bytes.Buffer{})
	if !errors.Is(err, ErrUnsupportedMetadata) {
		t.Errorf("ownership reject: expected ErrUnsupportedMetadata, got %v", err)
	}

	// Reject: special file type.
	noOwner := buildTarBytes(t, []testEntry{{name: "pipe", fifo: true}})
	_, err = (TranscodeArchive{Source: Tar{}, Target: Zip{}, Metadata: MetadataReject}).Transcode(
		context.Background(), bytes.NewReader(noOwner), &bytes.Buffer{})
	if !errors.Is(err, ErrUnsupportedMetadata) {
		t.Errorf("fifo reject: expected ErrUnsupportedMetadata, got %v", err)
	}

	// Preserve: special entry skipped, result explicitly incomplete, regular entries fine.
	res, out := transcode(t, TranscodeArchive{Source: Tar{}, Target: Zip{}, Metadata: MetadataPreserve}, noOwner)
	_ = res
	entries := parseZip(t, out)
	if _, ok := entries["pipe"]; ok {
		t.Error("fifo should be skipped under best-effort preserve")
	}

	// TAR target can hold the fifo natively.
	_, out = transcode(t, TranscodeArchive{Source: Tar{}, Target: Tar{}, Metadata: MetadataReject}, noOwner)
	tarEntries := parseTar(t, out)
	if tarEntries["pipe"].typeflag != tar.TypeFifo {
		t.Errorf("tar fifo not preserved: %+v", tarEntries["pipe"])
	}
}

func TestTranscode_UnsupportedCombinations(t *testing.T) {
	if _, err := (TranscodeArchive{Source: fakeExtractor{}, Target: Tar{}}).Transcode(
		context.Background(), nil, &bytes.Buffer{}); !errors.Is(err, ErrTranscodeUnsupportedCombo) {
		t.Errorf("fake source: expected ErrTranscodeUnsupportedCombo, got %v", err)
	}
	// Zip source needs ReaderAt+Seeker; a plain buffer is neither.
	if _, err := (TranscodeArchive{Source: Zip{}, Target: Tar{}}).Transcode(
		context.Background(), &bytes.Buffer{}, &bytes.Buffer{}); err == nil {
		t.Error("expected error for non-seekable Zip input")
	}
}

func TestTranscode_ConcurrencyBoundedAndOrdered(t *testing.T) {
	baseline := runtime.NumGoroutine()
	entries := []testEntry{
		{name: "e0", body: "0"}, {name: "e1", body: "1"}, {name: "e2", body: "2"},
		{name: "e3", body: "3"}, {name: "e4", body: "4"}, {name: "e5", body: "5"},
	}
	target := &holdingTarget{gate: make(chan struct{}, 1)}
	ta := TranscodeArchive{
		Source:         fakeExtractor{entries: entries},
		Target:         target,
		Hardlinks:      LinkMaterialize, // forces spool-backed, pipelinable jobs
		MaxConcurrency: 3,
	}

	done := make(chan error, 1)
	go func() {
		_, err := ta.run(context.Background(), nil, &bytes.Buffer{})
		done <- err
	}()

	// Release one job at a time slowly so concurrency can build up.
	go func() {
		for i := 0; i < len(entries); i++ {
			time.Sleep(30 * time.Millisecond)
			target.gate <- struct{}{}
		}
	}()

	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("run: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("pipelined run deadlocked")
	}

	if target.maxSeen.Load() > 3 {
		t.Errorf("max in-flight jobs=%d exceeds MaxConcurrency=3", target.maxSeen.Load())
	}
	if target.maxSeen.Load() < 2 {
		t.Errorf("expected pipelining beyond a single in-flight job, max=%d", target.maxSeen.Load())
	}
	wantOrder := []string{"e0", "e1", "e2", "e3", "e4", "e5"}
	if fmt.Sprint(target.order) != fmt.Sprint(wantOrder) {
		t.Errorf("entry order changed: got %v want %v", target.order, wantOrder)
	}
	assertGoroutinesSettled(t, baseline+1)
}

func mapKeys(m map[string]parsedEntry) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}
