package archives

import (
	"archive/tar"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"math/rand"
	"os"
	"path"
	"slices"
	"strings"
	"sync"
	"testing"
	"testing/fstest"
	"time"
)

// ---- source instrumentation -----------------------------------------------

// countingReaderAt records every ReadAt call and the number of bytes
// requested. It is the evidence that random-access serving never re-scans
// the archive: SectionReader issues exactly one ReadAt per logical read and
// never reads outside the requested body range.
type countingReaderAt struct {
	r     io.ReaderAt
	mu    sync.Mutex
	calls int64
	bytes int64
}

func (c *countingReaderAt) ReadAt(p []byte, off int64) (int, error) {
	c.mu.Lock()
	c.calls++
	c.bytes += int64(len(p))
	c.mu.Unlock()
	return c.r.ReadAt(p, off)
}

func (c *countingReaderAt) reset() {
	c.mu.Lock()
	c.calls, c.bytes = 0, 0
	c.mu.Unlock()
}

func (c *countingReaderAt) snapshot() (calls, read int64) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.calls, c.bytes
}

// blockingReaderAt fails the test if the source is ever read. It proves that
// a corrupt index is rejected before any source range is touched.
type blockingReaderAt struct{ t *testing.T }

func (b *blockingReaderAt) ReadAt(p []byte, off int64) (int, error) {
	b.t.Helper()
	b.t.Errorf("source must not be read for a rejected index: ReadAt(off=%d, len=%d)", off, len(p))
	return 0, errors.New("source access blocked")
}

// flipReaderAt flips one source byte to simulate post-index tampering.
type flipReaderAt struct {
	r       io.ReaderAt
	at      int64
	stopped bool
}

func (f *flipReaderAt) ReadAt(p []byte, off int64) (int, error) {
	n, err := f.r.ReadAt(p, off)
	if !f.stopped && f.at >= off && f.at < off+int64(n) {
		p[f.at-off] ^= 0xFF
	}
	return n, err
}

// ---- tar writing helpers ---------------------------------------------------

type expectedEntry struct {
	typeflag byte
	mode     fs.FileMode
	size     int64
	content  []byte
	linkname string
}

func writeHeader(t *testing.T, tw *tar.Writer, hdr *tar.Header, body []byte) {
	t.Helper()
	if err := tw.WriteHeader(hdr); err != nil {
		t.Fatalf("WriteHeader(%q): %v", hdr.Name, err)
	}
	if len(body) > 0 {
		if _, err := tw.Write(body); err != nil {
			t.Fatalf("Write body of %q: %v", hdr.Name, err)
		}
	}
}

func regHeader(name string, content []byte) *tar.Header {
	return &tar.Header{Name: name, Mode: 0o640, Size: int64(len(content)), Typeflag: tar.TypeReg, ModTime: time.Unix(1_700_000_000, 0)}
}

func dirHeader(name string) *tar.Header {
	return &tar.Header{Name: name, Mode: 0o755, Typeflag: tar.TypeDir, ModTime: time.Unix(1_700_000_000, 0)}
}

// putOctal writes v as width-1 zero-padded octal digits followed by NUL.
func putOctal(field []byte, v int64, width int) {
	s := fmt.Sprintf("%0*o", width-1, v)
	copy(field, s)
	field[width-1] = 0
}

// rawHeaderBlock builds a 512-byte ustar header block by hand (used where
// archive/tar.Writer exposes no public API: global PAX, GNU sparse flag).
func rawHeaderBlock(name, linkname string, mode, size int64, typ byte) [512]byte {
	var blk [512]byte
	copy(blk[0:100], name)
	putOctal(blk[100:108], mode, 8)
	putOctal(blk[108:116], 0, 8)
	putOctal(blk[116:124], 0, 8)
	putOctal(blk[124:136], size, 12)
	putOctal(blk[136:148], time.Unix(1_700_000_000, 0).Unix(), 12)
	for i := 148; i < 156; i++ {
		blk[i] = ' '
	}
	blk[156] = typ
	copy(blk[157:257], linkname)
	copy(blk[257:263], "ustar\x00")
	copy(blk[263:265], "00")
	var sum int
	for _, b := range blk {
		sum += int(b)
	}
	copy(blk[148:156], fmt.Sprintf("%06o\x00 ", sum))
	return blk
}

// paxRecord encodes one PAX record with the self-describing length prefix.
func paxRecord(key, value string) []byte {
	record := fmt.Sprintf("0 %s=%s\n", key, value)
	for {
		candidate := fmt.Sprintf("%d %s=%s\n", len(record), key, value)
		if len(candidate) == len(record) {
			return []byte(candidate)
		}
		record = candidate
	}
}

// writeRawPAXGlobal emits a TypeXGlobalHeader plus its padded body before
// the tar.Writer takes over. archive/tar applies these records to every
// subsequent entry.
func writeRawPAXGlobal(t *testing.T, w io.Writer, records map[string]string) {
	t.Helper()
	var body bytes.Buffer
	keys := make([]string, 0, len(records))
	for k := range records {
		keys = append(keys, k)
	}
	slices.Sort(keys)
	for _, k := range keys {
		body.Write(paxRecord(k, records[k]))
	}
	blk := rawHeaderBlock("globalhead", "", 0, int64(body.Len()), tar.TypeXGlobalHeader)
	if _, err := w.Write(blk[:]); err != nil {
		t.Fatal(err)
	}
	if _, err := w.Write(body.Bytes()); err != nil {
		t.Fatal(err)
	}
	if rem := body.Len() % 512; rem != 0 {
		if _, err := w.Write(make([]byte, 512-rem)); err != nil {
			t.Fatal(err)
		}
	}
}

// independentTarEntries walks the tar with the standard library alone, which
// is the oracle the indexed file system is cross-checked against.
func independentTarEntries(t *testing.T, data []byte) map[string]expectedEntry {
	t.Helper()
	tr := tar.NewReader(bytes.NewReader(data))
	out := make(map[string]expectedEntry)
	for {
		hdr, err := tr.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatalf("independent tar walk failed: %v", err)
		}
		if hdr.Typeflag == tar.TypeXGlobalHeader {
			continue
		}
		body, err := io.ReadAll(tr)
		if err != nil {
			t.Fatalf("independent tar read of %q: %v", hdr.Name, err)
		}
		name := path.Clean(hdr.Name)
		if name == "." {
			continue
		}
		out[name] = expectedEntry{
			typeflag: hdr.Typeflag,
			mode:     hdr.FileInfo().Mode(),
			size:     hdr.Size,
			content:  body,
			linkname: hdr.Linkname,
		}
	}
	return out
}

// buildComprehensiveTar assembles an archive exercising every framing
// feature the index must handle: global PAX, local PAX, GNU long names,
// padding at every size boundary, implicit directories, symlinks and
// forward-referenced hardlinks.
func buildComprehensiveTar(t *testing.T) (*bytes.Buffer, map[string][]byte) {
	t.Helper()
	var buf bytes.Buffer
	writeRawPAXGlobal(t, &buf, map[string]string{"MYAPP.global": "present"})

	tw := tar.NewWriter(&buf)
	contents := make(map[string][]byte)

	put := func(hdr *tar.Header, body []byte) {
		writeHeader(t, tw, hdr, body)
		if hdr.Typeflag == tar.TypeReg {
			contents[path.Clean(hdr.Name)] = body
		}
	}

	readme := []byte("top level file\n")
	put(regHeader("readme.txt", readme), readme)

	// Bodies around 512-byte padding boundaries.
	for _, size := range []int{0, 1, 511, 512, 513, 1024, 1025} {
		body := bytes.Repeat([]byte{byte(size)}, size)
		put(regHeader(fmt.Sprintf("pad/%d.bin", size), body), body)
	}

	// Explicit directory.
	put(dirHeader("logs/"), nil)
	logLine := []byte("log line\n")
	put(regHeader("logs/today.log", logLine), logLine)

	// Implicit directories: no entries for "imp" or "imp/deep".
	nested := []byte("nested\n")
	put(regHeader("imp/deep/file.txt", nested), nested)
	nestedOther := []byte("nested other\n")
	put(regHeader("imp/deep/other.txt", nestedOther), nestedOther)
	impTop := []byte("imp top\n")
	put(regHeader("imp/top.txt", impTop), impTop)

	// Relative symlink escaping and re-entering directories.
	put(&tar.Header{
		Name: "ln/readme", Typeflag: tar.TypeSymlink, Linkname: "../readme.txt",
		Mode: 0o777, ModTime: time.Unix(1_700_000_000, 0),
	}, nil)

	// Forward hardlink: link appears before its target.
	put(&tar.Header{
		Name: "links/hard", Typeflag: tar.TypeLink, Linkname: "links/orig",
		Mode: 0o640, ModTime: time.Unix(1_700_000_000, 0),
	}, nil)
	orig := []byte("shared hardlink content\n")
	put(regHeader("links/orig", orig), orig)

	// GNU long name (>100 bytes) forces TypeGNULongName framing.
	gnuBody := []byte("gnu long name body")
	gnuName := "gnu/" + strings.Repeat("g", 140) + ".dat"
	hdr := regHeader(gnuName, gnuBody)
	hdr.Format = tar.FormatGNU
	put(hdr, gnuBody)

	// PAX long path plus a custom local PAX record.
	paxBody := []byte("pax long name body")
	paxName := "pax/" + strings.Repeat("p", 130) + ".dat"
	hdr = regHeader(paxName, paxBody)
	hdr.Format = tar.FormatPAX
	hdr.PAXRecords = map[string]string{"MYAPP.custom": "value"}
	put(hdr, paxBody)

	if err := tw.Close(); err != nil {
		t.Fatal(err)
	}
	return &buf, contents
}

// ---- main cross-check test --------------------------------------------------

func TestIndexedTarFSCrossCheck(t *testing.T) {
	buf, writtenContents := buildComprehensiveTar(t)
	data := buf.Bytes()
	src := &countingReaderAt{r: bytes.NewReader(data)}

	idx, err := BuildTarIndex(context.Background(), src, int64(len(data)))
	if err != nil {
		t.Fatalf("BuildTarIndex: %v", err)
	}
	buildCalls, buildBytes := src.snapshot()
	if buildBytes != int64(len(data)) {
		t.Fatalf("indexing read %d bytes, want exactly %d (single scan)", buildBytes, len(data))
	}
	t.Logf("index scan: %d ReadAt calls, %d bytes for a %d-byte archive", buildCalls, buildBytes, len(data))

	expects := independentTarEntries(t, data)
	if len(expects) != len(idx.Entries) {
		t.Fatalf("entry count: index=%d want=%d", len(idx.Entries), len(expects))
	}

	fsys, err := NewIndexedTarFS(src, idx)
	if err != nil {
		t.Fatalf("NewIndexedTarFS: %v", err)
	}

	// Every expected entry exists with matching portable type and metadata.
	for name, want := range expects {
		info, err := fsys.Stat(name)
		if err != nil {
			t.Fatalf("Stat(%q): %v", name, err)
		}
		if info.Mode().Type() != want.mode.Type() {
			t.Errorf("Stat(%q) type = %v, want %v", name, info.Mode().Type(), want.mode.Type())
		}
		if info.Name() != path.Base(name) {
			t.Errorf("Stat(%q).Name() = %q, want %q", name, info.Name(), path.Base(name))
		}

		switch want.typeflag {
		case tar.TypeReg:
			got, err := fsys.ReadFile(name)
			if err != nil {
				t.Fatalf("ReadFile(%q): %v", name, err)
			}
			if !bytes.Equal(got, want.content) {
				t.Errorf("content of %q mismatch (got %d bytes, want %d)", name, len(got), len(want.content))
			}
			if written := writtenContents[name]; !bytes.Equal(got, written) {
				t.Errorf("content of %q differs from what was written", name)
			}
			if info.Size() != want.size {
				t.Errorf("Size(%q) = %d, want %d", name, info.Size(), want.size)
			}
		case tar.TypeLink:
			// Hardlink serves the target's bytes and reports target size.
			got, err := fsys.ReadFile(name)
			if err != nil {
				t.Fatalf("ReadFile(hardlink %q): %v", name, err)
			}
			target := path.Clean(want.linkname)
			if !bytes.Equal(got, expects[target].content) {
				t.Errorf("hardlink %q content does not match target %q", name, target)
			}
			if info.Size() != expects[target].size {
				t.Errorf("hardlink %q size = %d, want target size %d", name, info.Size(), expects[target].size)
			}
		case tar.TypeSymlink:
			gotTarget, err := fsys.ReadLink(name)
			if err != nil {
				t.Fatalf("ReadLink(%q): %v", name, err)
			}
			if gotTarget != want.linkname {
				t.Errorf("ReadLink(%q) = %q, want %q", name, gotTarget, want.linkname)
			}
			// Opening follows the link to its in-archive regular target.
			got, err := fsys.ReadFile(name)
			if err != nil {
				t.Fatalf("open symlink %q: %v", name, err)
			}
			resolved := path.Clean(path.Join(path.Dir(name), want.linkname))
			if !bytes.Equal(got, expects[resolved].content) {
				t.Errorf("symlink %q content does not match resolved target %q", name, resolved)
			}
		case tar.TypeDir:
			if !info.IsDir() {
				t.Errorf("expected %q to be a directory", name)
			}
		}
	}

	// Long names (GNU and PAX) survived extension framing.
	for prefix := range map[string]bool{"gnu/": true, "pax/": true} {
		found := false
		for name := range expects {
			if strings.HasPrefix(name, prefix) && len(name) > 100 {
				found = true
			}
		}
		if !found {
			t.Errorf("no %s long name entry indexed", prefix)
		}
	}

	// Directory listings, including implicit directories, match the
	// directory tree derived independently from the raw tar walk.
	wantDirs := map[string]map[string]struct{}{".": {}}
	for name := range expects {
		for parent, base := path.Dir(name), path.Base(name); ; parent, base = path.Dir(parent), path.Base(parent) {
			if wantDirs[parent] == nil {
				wantDirs[parent] = map[string]struct{}{}
			}
			wantDirs[parent][base] = struct{}{}
			if parent == "." {
				break
			}
		}
	}
	for dir, wantEntries := range wantDirs {
		got, err := fsys.ReadDir(dir)
		if err != nil {
			t.Fatalf("ReadDir(%q): %v", dir, err)
		}
		if len(got) != len(wantEntries) {
			t.Fatalf("ReadDir(%q) returned %d entries, want %d", dir, len(got), len(wantEntries))
		}
		for _, e := range got {
			if _, ok := wantEntries[e.Name()]; !ok {
				t.Errorf("ReadDir(%q): unexpected entry %q", dir, e.Name())
			}
		}
	}

	// Implicit directories are statable and openable.
	for _, implicit := range []string{"imp", "imp/deep"} {
		if _, err := fsys.Stat(implicit); err != nil {
			t.Errorf("implicit dir %q not statable: %v", implicit, err)
		}
		f, err := fsys.Open(implicit)
		if err != nil {
			t.Errorf("Open(%q): %v", implicit, err)
			continue
		}
		f.Close()
	}

	// Index roundtrip through JSON preserves all behavior, and the encoded
	// form contains no callback/path machinery (plain data contract).
	encoded, err := EncodeTarIndex(idx)
	if err != nil {
		t.Fatalf("EncodeTarIndex: %v", err)
	}
	if bytes.Contains(encoded, []byte("Open")) || bytes.Contains(encoded, []byte(".go")) {
		t.Errorf("encoded index appears to leak non-data fields")
	}
	decoded, err := DecodeTarIndex(encoded)
	if err != nil {
		t.Fatalf("DecodeTarIndex: %v", err)
	}
	fsys2, err := NewIndexedTarFS(src, decoded)
	if err != nil {
		t.Fatalf("NewIndexedTarFS from decoded index: %v", err)
	}
	for name := range writtenContents {
		a, err1 := fsys.ReadFile(name)
		b, err2 := fsys2.ReadFile(name)
		if err1 != nil || err2 != nil || !bytes.Equal(a, b) {
			t.Errorf("content mismatch after index roundtrip for %q", name)
		}
	}
}

// ---- large archive random access and source-read proof ----------------------

func TestIndexedTarFSLargeRandomAccess(t *testing.T) {
	const fileCount = 250
	rng := rand.New(rand.NewSource(42))
	var buf bytes.Buffer
	tw := tar.NewWriter(&buf)
	want := make([][]byte, fileCount)
	names := make([]string, fileCount)
	for i := 0; i < fileCount; i++ {
		size := rng.Intn(3000) // spans many 512-byte padding boundaries
		body := make([]byte, size)
		rng.Read(body)
		name := fmt.Sprintf("data/%04d.bin", i)
		writeHeader(t, tw, regHeader(name, body), body)
		want[i] = body
		names[i] = name
	}
	if err := tw.Close(); err != nil {
		t.Fatal(err)
	}
	data := buf.Bytes()

	src := &countingReaderAt{r: bytes.NewReader(data)}
	idx, err := BuildTarIndex(context.Background(), src, int64(len(data)))
	if err != nil {
		t.Fatalf("BuildTarIndex: %v", err)
	}
	_, scanBytes := src.snapshot()
	var bodySum int64
	for _, b := range want {
		bodySum += int64(len(b))
	}
	t.Logf("archive=%d bytes; one indexing scan read=%d bytes; sum of file bodies=%d bytes",
		len(data), scanBytes, bodySum)
	if scanBytes != int64(len(data)) || bodySum >= scanBytes {
		t.Fatalf("scan accounting wrong: scan=%d archive=%d bodies=%d", scanBytes, len(data), bodySum)
	}

	fsys, err := NewIndexedTarFS(src, idx)
	if err != nil {
		t.Fatal(err)
	}
	src.reset()

	// Random ReadAt traffic: each logical read must map to exactly one
	// source ReadAt covering only the requested byte range.
	const ops = 500
	var requested int64
	for i := 0; i < ops; i++ {
		fileIdx := rng.Intn(fileCount)
		body := want[fileIdx]
		off := int64(0)
		length := rng.Intn(len(body) + 1)
		if len(body) > 0 {
			off = int64(rng.Intn(len(body) + 1))
			if int(off)+length > len(body) {
				length = len(body) - int(off)
			}
		} else {
			length = 0
		}
		p := make([]byte, length)
		f, err := fsys.Open(names[fileIdx])
		if err != nil {
			t.Fatalf("Open(%s): %v", names[fileIdx], err)
		}
		rat := f.(io.ReaderAt)
		n, err := rat.ReadAt(p, off)
		f.Close()
		if length > 0 && err != nil && !errors.Is(err, io.EOF) {
			t.Fatalf("ReadAt(%s, off=%d, len=%d): %v", names[fileIdx], off, length, err)
		}
		if !bytes.Equal(p[:n], body[off:off+int64(n)]) {
			t.Fatalf("random read mismatch in %s at %d", names[fileIdx], off)
		}
		requested += int64(length)
	}
	calls, readBytes := src.snapshot()
	t.Logf("random serving: %d logical reads -> %d source ReadAt calls, %d bytes requested / %d source bytes read",
		ops, calls, requested, readBytes)
	if calls != ops {
		t.Errorf("source ReadAt calls = %d, want exactly %d (no rescans)", calls, ops)
	}
	if readBytes != requested {
		t.Errorf("source bytes read = %d, want %d (only requested ranges)", readBytes, requested)
	}

	// Opening and fully reading every file touches only body bytes, never
	// header blocks, padding or unrelated entries.
	src.reset()
	for i, name := range names {
		got, err := fsys.ReadFile(name)
		if err != nil {
			t.Fatalf("ReadFile(%s): %v", name, err)
		}
		if !bytes.Equal(got, want[i]) {
			t.Fatalf("full read mismatch in %s", name)
		}
	}
	_, fullRead := src.snapshot()
	t.Logf("full serving of %d files read %d source bytes vs %d archive bytes (%.1f%% of the archive; headers/padding never read)",
		fileCount, fullRead, len(data), 100*float64(fullRead)/float64(len(data)))
	if fullRead != bodySum {
		t.Errorf("serving all files read %d bytes, want %d (body bytes only)", fullRead, bodySum)
	}
}

// ---- ReaderAt EOF contract and independent handles --------------------------

func TestIndexedTarFileReaderAtContract(t *testing.T) {
	body := bytes.Repeat([]byte("abcdefgh"), 64) // 512 bytes + 1 = 513 below
	body = append(body, 'Z')                     // 513 bytes, exercises padding
	var buf bytes.Buffer
	tw := tar.NewWriter(&buf)
	writeHeader(t, tw, regHeader("f.bin", body), body)
	tw.Close()
	data := buf.Bytes()

	idx, err := BuildTarIndex(context.Background(), bytes.NewReader(data), int64(len(data)))
	if err != nil {
		t.Fatal(err)
	}
	fsys, err := NewIndexedTarFS(bytes.NewReader(data), idx)
	if err != nil {
		t.Fatal(err)
	}

	f, err := fsys.Open("f.bin")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	rat := f.(io.ReaderAt)

	// Read at/after EOF must return (0, io.EOF).
	for _, off := range []int64{int64(len(body)), int64(len(body)) + 1, int64(len(body)) + 4096} {
		p := make([]byte, 16)
		n, err := rat.ReadAt(p, off)
		if n != 0 || !errors.Is(err, io.EOF) {
			t.Errorf("ReadAt(off=%d) = (%d, %v), want (0, EOF)", off, n, err)
		}
	}

	// A read spanning EOF returns the available bytes and io.EOF.
	p := make([]byte, 32)
	n, err := rat.ReadAt(p, int64(len(body))-4)
	if n != 4 || !errors.Is(err, io.EOF) || !bytes.Equal(p[:4], body[len(body)-4:]) {
		t.Errorf("spanning ReadAt = (%d, %v), want (4, EOF) with tail bytes", n, err)
	}

	// Two handles keep independent offsets; ReadAt ignores shared state and
	// Seek only affects the handle it was called on.
	f2, err := fsys.Open("f.bin")
	if err != nil {
		t.Fatal(err)
	}
	defer f2.Close()
	seeker1 := f.(io.Seeker)
	if _, err := seeker1.Seek(100, io.SeekStart); err != nil {
		t.Fatal(err)
	}
	buf1 := make([]byte, 5)
	if _, err := f2.Read(buf1); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(buf1, body[0:5]) {
		t.Errorf("second handle offset was affected by first handle Seek: got %q want %q", buf1, body[0:5])
	}
	buf2 := make([]byte, 5)
	if _, err := f.Read(buf2); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(buf2, body[100:105]) {
		t.Errorf("first handle lost its Seek position: got %q want %q", buf2, body[100:105])
	}
	if n, err := rat.ReadAt(buf2, 7); err != nil || !bytes.Equal(buf2, body[7:12]) {
		t.Errorf("ReadAt after Seek = (%d, %v), want absolute offset 7", n, err)
	}
}

// ---- hardlink and symlink graph failures ------------------------------------

func writeLinkOnlyTar(t *testing.T, entries []struct{ name, target string; typ byte }) []byte {
	var buf bytes.Buffer
	tw := tar.NewWriter(&buf)
	for _, e := range entries {
		writeHeader(t, tw, &tar.Header{
			Name: e.name, Typeflag: e.typ, Linkname: e.target,
			Mode: 0o640, ModTime: time.Unix(1_700_000_000, 0),
		}, nil)
	}
	tw.Close()
	return buf.Bytes()
}

func TestHardlinkGraphFailures(t *testing.T) {
	cycle := writeLinkOnlyTar(t, []struct{ name, target string; typ byte }{
		{"a", "b", tar.TypeLink},
		{"b", "a", tar.TypeLink},
	})
	if _, err := BuildTarIndex(context.Background(), bytes.NewReader(cycle), int64(len(cycle))); !errors.Is(err, ErrHardlinkCycle) {
		t.Errorf("cycle: got %v, want ErrHardlinkCycle", err)
	}

	dangling := writeLinkOnlyTar(t, []struct{ name, target string; typ byte }{
		{"x", "missing", tar.TypeLink},
	})
	if _, err := BuildTarIndex(context.Background(), bytes.NewReader(dangling), int64(len(dangling))); !errors.Is(err, ErrHardlinkTarget) {
		t.Errorf("dangling: got %v, want ErrHardlinkTarget", err)
	}

	// Symlink cycles are legal in the index but fail when opened.
	sym := writeLinkOnlyTar(t, []struct{ name, target string; typ byte }{
		{"s1", "s2", tar.TypeSymlink},
		{"s2", "s1", tar.TypeSymlink},
	})
	idx, err := BuildTarIndex(context.Background(), bytes.NewReader(sym), int64(len(sym)))
	if err != nil {
		t.Fatalf("symlink cycle should index fine, got %v", err)
	}
	fsys, err := NewIndexedTarFS(bytes.NewReader(sym), idx)
	if err != nil {
		t.Fatal(err)
	}
	if target, err := fsys.ReadLink("s1"); err != nil || target != "s2" {
		t.Errorf("ReadLink(s1) = %q, %v", target, err)
	}
	if _, err := fsys.Open("s1"); !errors.Is(err, ErrHardlinkCycle) {
		t.Errorf("open symlink cycle: got %v, want ErrHardlinkCycle", err)
	}

	// Dangling symlink opens with NotExist.
	danglingSym := writeLinkOnlyTar(t, []struct{ name, target string; typ byte }{
		{"s", "nowhere", tar.TypeSymlink},
	})
	idx2, err := BuildTarIndex(context.Background(), bytes.NewReader(danglingSym), int64(len(danglingSym)))
	if err != nil {
		t.Fatal(err)
	}
	fsys2, err := NewIndexedTarFS(bytes.NewReader(danglingSym), idx2)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := fsys2.Open("s"); !errors.Is(err, fs.ErrNotExist) {
		t.Errorf("open dangling symlink: got %v, want fs.ErrNotExist", err)
	}
}

// ---- sparse archives are explicitly rejected --------------------------------

func TestSparseTarRejected(t *testing.T) {
	// Hand-craft a PAX local header carrying GNU.sparse.* records (the
	// standard tar.Writer drops these keys) followed by the described file.
	// archive/tar recognizes the sparse map and reconstructs logical data;
	// the index must refuse up front instead of exposing the stored bytes
	// through a contiguous SectionReader as if there were no holes.
	var paxBody bytes.Buffer
	for _, rec := range [][2]string{
		// GNU sparse 0.1: numblocks + comma-separated offset,length pairs.
		{"GNU.sparse.major", "0"},
		{"GNU.sparse.minor", "1"},
		{"GNU.sparse.numblocks", "1"},
		{"GNU.sparse.map", "0,10"},
		{"GNU.sparse.size", "10"},
		{"path", "sparse.dat"},
		{"size", "10"},
	} {
		paxBody.Write(paxRecord(rec[0], rec[1]))
	}
	var buf bytes.Buffer
	paxHdr := rawHeaderBlock("pax.local", "", 0o644, int64(paxBody.Len()), tar.TypeXHeader)
	buf.Write(paxHdr[:])
	buf.Write(paxBody.Bytes())
	if rem := paxBody.Len() % 512; rem != 0 {
		buf.Write(make([]byte, 512-rem))
	}
	fileHdr := rawHeaderBlock("sparse.dat", "", 0o644, 10, tar.TypeReg)
	buf.Write(fileHdr[:])
	buf.Write(bytes.Repeat([]byte{'x'}, 10))
	buf.Write(make([]byte, 502)) // pad body to a block
	buf.Write(make([]byte, 1024))

	// Sanity check: archive/tar itself parses this as a sparse file.
	tr := tar.NewReader(bytes.NewReader(buf.Bytes()))
	hdr, err := tr.Next()
	if err != nil {
		t.Fatalf("archive/tar failed to parse crafted sparse tar: %v", err)
	}
	if !tarHeaderIsSparse(hdr) {
		t.Fatalf("test setup: crafted header not recognized as sparse")
	}

	if _, err := BuildTarIndex(context.Background(), bytes.NewReader(buf.Bytes()), int64(buf.Len())); !errors.Is(err, ErrSparseTarUnsupported) {
		t.Errorf("PAX sparse: got %v, want ErrSparseTarUnsupported", err)
	}

	// Raw GNU sparse type flag: indexing must fail one way or another rather
	// than silently exposing hole bytes as file content.
	var raw bytes.Buffer
	blk := rawHeaderBlock("oldsparse", "", 0o644, 10, tar.TypeGNUSparse)
	raw.Write(blk[:])
	raw.Write(bytes.Repeat([]byte{0}, 10))
	raw.Write(make([]byte, 502)) // pad to 512
	raw.Write(make([]byte, 1024))
	if _, err := BuildTarIndex(context.Background(), bytes.NewReader(raw.Bytes()), int64(raw.Len())); err == nil {
		t.Error("raw GNU sparse archive was indexed without error")
	}
}

// ---- corrupt and tampered indexes -------------------------------------------

func TestCorruptIndexRejectedBeforeSourceAccess(t *testing.T) {
	buf, _ := buildComprehensiveTar(t)
	data := buf.Bytes()
	good, err := BuildTarIndex(context.Background(), bytes.NewReader(data), int64(len(data)))
	if err != nil {
		t.Fatal(err)
	}

	cases := map[string]func(*TarIndex){
		"bad version": func(i *TarIndex) { i.Version = 99 },
		"out of bounds body": func(i *TarIndex) {
			i.Entries[len(i.Entries)-1].BodyOff = i.Size + 512
		},
		"overlapping ranges": func(i *TarIndex) {
			i.Entries[1].HeaderOff = i.Entries[0].HeaderOff
			i.Entries[1].BodyOff = i.Entries[0].BodyOff
		},
		"duplicate names": func(i *TarIndex) {
			i.Entries[1].Name = i.Entries[0].Name
		},
		"bad digest length": func(i *TarIndex) { i.Digest = []byte{1, 2, 3} },
		"unaligned offset": func(i *TarIndex) { i.Entries[0].BodyOff += 1 },
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			idx := *good
			idx.Entries = append([]TarEntry(nil), good.Entries...)
			idx.byName, idx.dirs = nil, nil
			mutate(&idx)

			// Serialized form must also be rejected on decode.
			encoded, encErr := json.Marshal(&idx)
			if encErr != nil {
				t.Fatal(encErr)
			}
			if _, err := DecodeTarIndex(encoded); err == nil {
				t.Errorf("DecodeTarIndex accepted tampered index (%s)", name)
			}

			// FS construction must refuse and never touch the source.
			if _, err := NewIndexedTarFS(&blockingReaderAt{t: t}, &idx); err == nil {
				t.Errorf("NewIndexedTarFS accepted tampered index (%s)", name)
			}
		})
	}
}

func TestDigestVerification(t *testing.T) {
	buf, _ := buildComprehensiveTar(t)
	data := buf.Bytes()
	idx, err := BuildTarIndex(context.Background(), bytes.NewReader(data), int64(len(data)))
	if err != nil {
		t.Fatal(err)
	}

	// Explicit full-digest verification succeeds on the matching source.
	if _, err := NewIndexedTarFS(bytes.NewReader(data), idx, WithVerifyDigest(true)); err != nil {
		t.Fatalf("verification of unchanged source failed: %v", err)
	}

	// One flipped source byte must fail verification, and verification is
	// the only extra full-source read performed at open.
	flipped := &flipReaderAt{r: bytes.NewReader(data), at: int64(len(data) / 2)}
	if _, err := NewIndexedTarFS(flipped, idx, WithVerifyDigest(true)); !errors.Is(err, ErrCorruptTarIndex) {
		t.Errorf("tampered source: got %v, want ErrCorruptTarIndex", err)
	}

	// Without explicit verification the structural index still opens even
	// though the bytes changed: digest identity is opt-in, never assumed
	// from size or mtime.
	fsys, err := NewIndexedTarFS(flipped, idx)
	if err != nil {
		t.Fatalf("open without verification failed: %v", err)
	}
	if _, err := fsys.Stat("."); err != nil {
		t.Errorf("stat after tamper without verification: %v", err)
	}
}

// ---- duplicate names and path normalization ---------------------------------

func TestDuplicateAndPathPolicy(t *testing.T) {
	build := func(names [2]string, bodies [2][]byte) []byte {
		var buf bytes.Buffer
		tw := tar.NewWriter(&buf)
		writeHeader(t, tw, regHeader(names[0], bodies[0]), bodies[0])
		writeHeader(t, tw, regHeader(names[1], bodies[1]), bodies[1])
		tw.Close()
		return buf.Bytes()
	}

	// "a//b" and "a/./b" both normalize to "a/b": a normalization collision.
	data := build([2]string{"a/b", "a//b"}, [2][]byte{[]byte("first"), []byte("second")})
	if _, err := BuildTarIndex(context.Background(), bytes.NewReader(data), int64(len(data)), WithTarDuplicatePolicy(TarDuplicateError)); err == nil {
		t.Error("strict policy should reject duplicate logical names")
	}
	for _, tc := range []struct {
		policy TarDuplicatePolicy
		want   string
	}{
		{TarDuplicateKeepFirst, "first"},
		{TarDuplicateKeepLast, "second"},
	} {
		idx, err := BuildTarIndex(context.Background(), bytes.NewReader(data), int64(len(data)), WithTarDuplicatePolicy(tc.policy))
		if err != nil {
			t.Fatalf("policy %d: %v", tc.policy, err)
		}
		fsys, err := NewIndexedTarFS(bytes.NewReader(data), idx)
		if err != nil {
			t.Fatal(err)
		}
		got, err := fsys.ReadFile("a/b")
		if err != nil || string(got) != tc.want {
			t.Errorf("policy %d: got %q (%v), want %q", tc.policy, got, err, tc.want)
		}
	}

	// Unsafe paths are rejected rather than normalized into the root.
	for _, bad := range []string{"/etc/passwd", "../escape", "a/../../b"} {
		var buf bytes.Buffer
		tw := tar.NewWriter(&buf)
		writeHeader(t, tw, regHeader(bad, []byte("x")), []byte("x"))
		tw.Close()
		if _, err := BuildTarIndex(context.Background(), bytes.NewReader(buf.Bytes()), int64(buf.Len())); err == nil {
			t.Errorf("unsafe path %q was accepted", bad)
		}
	}
}

// ---- Sub, sticky error root and close behavior -------------------------------

func TestIndexedTarFSSubAndClose(t *testing.T) {
	buf, _ := buildComprehensiveTar(t)
	data := buf.Bytes()
	src := &countingReaderAt{r: bytes.NewReader(data)}
	idx, err := BuildTarIndex(context.Background(), src, int64(len(data)))
	if err != nil {
		t.Fatal(err)
	}
	fsys, err := NewIndexedTarFS(src, idx)
	if err != nil {
		t.Fatal(err)
	}

	sub, err := fsys.Sub("imp")
	if err != nil {
		t.Fatalf("Sub: %v", err)
	}
	entries, err := fs.ReadDir(sub, "deep")
	if err != nil {
		t.Fatalf("ReadDir through Sub: %v", err)
	}
	if len(entries) != 2 {
		t.Errorf("Sub ReadDir(deep) = %d entries, want 2", len(entries))
	}
	got, err := fs.ReadFile(sub, "deep/file.txt")
	if err != nil || string(got) != "nested\n" {
		t.Errorf("ReadFile through Sub = %q, %v", got, err)
	}
	if _, err := fsys.Sub("readme.txt"); err == nil {
		t.Error("Sub on a regular file should fail")
	}
	if _, err := fsys.Sub("missing"); err == nil {
		t.Error("Sub on missing path should fail")
	}
	if _, err := fsys.Sub("invalid/path/"); err == nil {
		t.Error("Sub with invalid path should fail")
	}

	// Sub layers prefixes without resetting a sticky construction error.
	sentinel := errors.New("boom")
	broken := &IndexedTarFS{source: src, index: idx, rootErr: sentinel}
	brokenSub, err := broken.Sub(".")
	if err != nil {
		t.Fatalf("Sub with rootErr: %v", err)
	}
	if _, err := brokenSub.Open("readme.txt"); !errors.Is(err, sentinel) {
		t.Errorf("Open after Sub = %v, want preserved rootErr", err)
	}
	if _, err := brokenSub.(*IndexedTarFS).ReadDir("."); !errors.Is(err, sentinel) {
		t.Errorf("ReadDir after Sub = %v, want preserved rootErr", err)
	}

	// Close must not close the caller-owned source: it stays readable and
	// the FS itself reports no error.
	if err := fsys.Close(); err != nil {
		t.Errorf("Close = %v", err)
	}
	probe := make([]byte, 1)
	if _, err := src.ReadAt(probe, 0); err != nil {
		t.Errorf("caller source unusable after IndexedTarFS.Close: %v", err)
	}
}

// ---- standard library fs.FS conformance --------------------------------------

func TestIndexedTarFSFstestConformance(t *testing.T) {
	var buf bytes.Buffer
	tw := tar.NewWriter(&buf)
	writeHeader(t, tw, dirHeader("a/"), nil)
	one, two, top, deep := []byte("one"), []byte("two"), []byte("top"), []byte("deep")
	writeHeader(t, tw, regHeader("a/one.txt", one), one)
	writeHeader(t, tw, regHeader("a/two.txt", two), two)
	writeHeader(t, tw, regHeader("top.txt", top), top)
	writeHeader(t, tw, regHeader("implicit/deep/file.bin", deep), deep)
	tw.Close()
	data := buf.Bytes()

	idx, err := BuildTarIndex(context.Background(), bytes.NewReader(data), int64(len(data)))
	if err != nil {
		t.Fatal(err)
	}
	fsys, err := NewIndexedTarFS(bytes.NewReader(data), idx)
	if err != nil {
		t.Fatal(err)
	}
	if err := fstest.TestFS(fsys, "a/one.txt", "a/two.txt", "top.txt", "implicit/deep/file.bin"); err != nil {
		t.Fatalf("fstest.TestFS: %v", err)
	}

	// Directory handle paging follows os.File semantics: repeated
	// ReadDir(-1) on one handle must not hand back the listing twice.
	root, err := fsys.Open(".")
	if err != nil {
		t.Fatal(err)
	}
	first, err := root.(fs.ReadDirFile).ReadDir(-1)
	if err != nil || len(first) == 0 {
		t.Fatalf("first ReadDir(-1): %d entries, %v", len(first), err)
	}
	second, err := root.(fs.ReadDirFile).ReadDir(-1)
	if err != nil || len(second) != 0 {
		t.Errorf("second ReadDir(-1) = %d entries, %v; want empty, nil", len(second), err)
	}
	root.Close()
}

// ---- real testdata archives cross-checked with archive/tar --------------------

func TestIndexedTarFSOnTestdata(t *testing.T) {
	for _, archive := range []string{"testdata/gnu-hardlinks.tar", "testdata/self-tar.tar"} {
		t.Run(archive, func(t *testing.T) {
			f, err := os.Open(archive)
			if err != nil {
				t.Fatal(err)
			}
			defer f.Close()
			info, err := f.Stat()
			if err != nil {
				t.Fatal(err)
			}
			idx, err := BuildTarIndex(context.Background(), f, info.Size())
			if err != nil {
				t.Fatalf("BuildTarIndex: %v", err)
			}
			fsys, err := NewIndexedTarFS(f, idx)
			if err != nil {
				t.Fatal(err)
			}

			// Independent walk of the same on-disk archive is the oracle.
			oracle, err := os.Open(archive)
			if err != nil {
				t.Fatal(err)
			}
			defer oracle.Close()
			tr := tar.NewReader(oracle)
			checked := 0
			for {
				hdr, err := tr.Next()
				if errors.Is(err, io.EOF) {
					break
				}
				if err != nil {
					t.Fatalf("oracle walk: %v", err)
				}
				if hdr.Typeflag == tar.TypeXGlobalHeader || path.Clean(hdr.Name) == "." {
					continue
				}
				name := path.Clean(hdr.Name)
				checked++

				st, err := fsys.Stat(name)
				if err != nil {
					t.Fatalf("Stat(%q): %v", name, err)
				}
				switch hdr.Typeflag {
				case tar.TypeReg, tar.TypeCont:
					want, rerr := io.ReadAll(tr)
					if rerr != nil {
						t.Fatal(rerr)
					}
					got, err := fsys.ReadFile(name)
					if err != nil {
						t.Fatalf("ReadFile(%q): %v", name, err)
					}
					if !bytes.Equal(got, want) {
						t.Errorf("content mismatch for %q", name)
					}
					if st.Size() != int64(len(want)) {
						t.Errorf("size mismatch for %q: %d vs %d", name, st.Size(), len(want))
					}
				case tar.TypeLink:
					got, err := fsys.ReadFile(name)
					if err != nil {
						t.Fatalf("hardlink %q: %v", name, err)
					}
					if len(got) == 0 && hdr.Size > 0 {
						t.Errorf("hardlink %q resolved to empty body", name)
					}
				}
			}
			if checked != len(idx.Entries) {
				t.Errorf("checked %d oracle entries vs %d indexed entries", checked, len(idx.Entries))
			}

			// Caller-owned source remains usable after Close.
			if err := fsys.Close(); err != nil {
				t.Errorf("Close: %v", err)
			}
			probe := make([]byte, 1)
			if _, err := f.ReadAt(probe, 0); err != nil {
				t.Errorf("source closed by IndexedTarFS: %v", err)
			}
		})
	}
}
