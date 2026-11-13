package archives

import (
	"archive/tar"
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"math/rand/v2"
	"strconv"
	"strings"
	"testing"
	"time"

	kzip "github.com/klauspost/compress/zip"
)

// ---- in-memory test doubles -------------------------------------------------

// fakeInode stands in for a host (dev, inode) pair. It is readable through the
// unexported hardlinkKeyFunc test seam on every platform, including Windows
// where no portable inode API is available.
type fakeInode struct {
	dev uint64
	ino uint64
}

type memFileInfo struct {
	name    string
	size    int64
	mode    fs.FileMode
	modTime time.Time
	sys     any
}

func (m memFileInfo) Name() string       { return m.name }
func (m memFileInfo) Size() int64        { return m.size }
func (m memFileInfo) Mode() fs.FileMode  { return m.mode }
func (m memFileInfo) ModTime() time.Time { return m.modTime }
func (m memFileInfo) IsDir() bool        { return m.mode.IsDir() }
func (m memFileInfo) Sys() any           { return m.sys }

type memEntry struct {
	path      string
	mode      fs.FileMode
	content   []byte // regular-file content (also the anchor content of hard links)
	target    string // symbolic-link target
	inode     *fakeInode
	openCount *int
	// declaredSize overrides the size advertised by FileInfo (default: len(content)).
	declaredSize int64
	// actualContent overrides what Open returns (default: content).
	actualContent []byte
	// hostHeader is assigned to FileInfo.Header to prove host headers never leak.
	hostHeader any
	// hostLinkTarget pre-fills LinkTarget, simulating FilesFromDisk first-walk behavior.
	hostLinkTarget string
}

func buildFiles(entries []memEntry) []FileInfo {
	files := make([]FileInfo, 0, len(entries))
	for _, e := range entries {
		e := e
		content := e.content
		if e.actualContent != nil {
			content = e.actualContent
		}
		declared := e.declaredSize
		if declared == 0 {
			declared = int64(len(e.content))
		}
		info := memFileInfo{
			name:    baseName(e.path),
			size:    declared,
			mode:    e.mode,
			modTime: time.Date(2001, 2, 3, 4, 5, 6, 0, time.UTC),
			sys:     e.inode,
		}
		var openCount *int
		if e.openCount != nil {
			openCount = e.openCount
		}
		linkTarget := e.hostLinkTarget
		if e.mode&fs.ModeSymlink != 0 {
			linkTarget = e.target // symbolic links carry their explicit target
		}
		files = append(files, FileInfo{
			FileInfo:      info,
			Header:        e.hostHeader,
			NameInArchive: e.path,
			LinkTarget:    linkTarget,
			Open: func() (fs.File, error) {
				if openCount != nil {
					*openCount++
				}
				return &memFile{Reader: bytes.NewReader(content)}, nil
			},
		})
	}
	return files
}

func baseName(p string) string {
	if i := strings.LastIndex(p, "/"); i >= 0 {
		return p[i+1:]
	}
	return p
}

type memFile struct {
	*bytes.Reader
}

func (memFile) Close() error { return nil }
func (memFile) Stat() (fs.FileInfo, error) {
	return nil, errors.New("memFile.Stat not implemented")
}

func fakeHardlinkKey(fi fs.FileInfo) (canonicalInodeKey, bool) {
	in, ok := fi.Sys().(*fakeInode)
	if !ok || in == nil {
		return canonicalInodeKey{}, false
	}
	return canonicalInodeKey{dev: in.dev, ino: in.ino}, true
}

func testCanonical() CanonicalArchive {
	return CanonicalArchive{
		Template: Tar{},
		Metadata: CanonicalMetadata{
			ModTime: time.Date(2020, 1, 2, 3, 4, 5, 0, time.UTC),
			Uid:     1000,
			Gid:     2000,
			Uname:   "alice",
			Gname:   "builders",
		},
		hardlinkKeyFunc: fakeHardlinkKey,
	}
}

func archiveBytes(t *testing.T, ca CanonicalArchive, files []FileInfo) []byte {
	t.Helper()
	var buf bytes.Buffer
	if err := ca.Archive(context.Background(), &buf, files); err != nil {
		t.Fatalf("Archive failed: %v", err)
	}
	return buf.Bytes()
}

// failOnWriteWriter must never receive a byte: used to prove validation
// failures reject output before anything is written.
type failOnWriteWriter struct{}

func (failOnWriteWriter) Write(p []byte) (int, error) {
	return 0, fmt.Errorf("test writer: unexpected write of %d bytes", len(p))
}

func shuffledCopy(rng *rand.Rand, files []FileInfo) []FileInfo {
	out := append([]FileInfo(nil), files...)
	rng.Shuffle(len(out), func(i, j int) { out[i], out[j] = out[j], out[i] })
	return out
}

// canonicalTree is a shared, deliberately non-sorted logical file set that
// includes a directory, regular files with different permissions, a symbolic
// link, a UTF-8 (non-ASCII) name, and a three-member hard-link set.
func canonicalTree() []memEntry {
	inode := &fakeInode{dev: 7, ino: 99}
	return []memEntry{
		{path: "zzz/last.txt", mode: 0o600, content: []byte("last file content")},
		{path: "dir", mode: fs.ModeDir | 0o755},
		{path: "dir/a.txt", mode: 0o644, content: []byte("alpha")},
		{path: "root.txt", mode: 0o640, content: []byte("root contents\n")},
		{path: "link-to-root", mode: fs.ModeSymlink | 0o777, target: "root.txt"},
		{path: "文件名-日本語.txt", mode: 0o644, content: []byte("unicode body")},
		{path: "z-hard.txt", mode: 0o644, content: []byte("shared hardlink body"), inode: inode, hostLinkTarget: "zzz/last.txt"},
		{path: "a-hard.txt", mode: 0o644, content: []byte("shared hardlink body"), inode: inode},
		{path: "m-hard.txt", mode: 0o644, content: []byte("shared hardlink body"), inode: inode},
	}
}

// hostVariants returns copies of the tree with perturbed host attributes:
// different mtimes, different (but equivalent) inode pointers, and pre-filled
// hard-link targets that disagree with the canonical anchor. None of these
// may affect the output bytes.
func hostVariants(base []memEntry) [][]memEntry {
	var variants [][]memEntry

	v1 := append([]memEntry(nil), base...)
	for i := range v1 {
		v1[i].mode = base[i].mode // permissions intentionally unchanged
	}
	variants = append(variants, v1)

	v2 := append([]memEntry(nil), base...)
	for i := range v2 {
		v2[i].mode = base[i].mode
		if v2[i].inode != nil {
			v2[i].inode = &fakeInode{dev: v2[i].inode.dev, ino: v2[i].inode.ino}
		}
	}
	variants = append(variants, v2)

	v3 := append([]memEntry(nil), base...)
	for i := range v3 {
		if v3[i].inode != nil {
			// Simulate FilesFromDisk having walked a non-anchor member first.
			v3[i].hostLinkTarget = "z-hard.txt"
		}
	}
	variants = append(variants, v3)

	return variants
}

// ---- TAR determinism --------------------------------------------------------

func TestCanonicalTarDeterministic(t *testing.T) {
	ca := testCanonical()
	baseFiles := buildFiles(canonicalTree())
	reference := archiveBytes(t, ca, baseFiles)

	// Re-archiving the identical logical set must reproduce every byte.
	if got := archiveBytes(t, ca, buildFiles(canonicalTree())); !bytes.Equal(got, reference) {
		t.Error("repeated canonical tar archiving produced different bytes")
	}

	// Random permutations of the input slice must not change a single byte.
	rng := rand.New(rand.NewPCG(1, 99))
	for i := 0; i < 8; i++ {
		got := archiveBytes(t, ca, shuffledCopy(rng, baseFiles))
		if !bytes.Equal(got, reference) {
			t.Fatalf("permutation %d changed the canonical tar bytes", i)
		}
	}

	// Perturbed host attributes must not change a single byte.
	for vi, variant := range hostVariants(canonicalTree()) {
		got := archiveBytes(t, ca, shuffledCopy(rng, buildFiles(variant)))
		if !bytes.Equal(got, reference) {
			t.Fatalf("host variant %d changed the canonical tar bytes", vi)
		}
	}

	verifyTarContents(t, reference, ca.Metadata)
}

func verifyTarContents(t *testing.T, data []byte, meta CanonicalMetadata) {
	t.Helper()
	tr := tar.NewReader(bytes.NewReader(data))

	seen := map[string]bool{}
	var prevName string
	first := true
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("tar.Next: %v", err)
		}

		// Entries must be emitted in sorted canonical order.
		if !first && hdr.Name < prevName {
			t.Errorf("entries not sorted: %q followed by %q", prevName, hdr.Name)
		}
		first = false
		prevName = hdr.Name
		seen[hdr.Name] = true

		if !hdr.ModTime.Equal(meta.ModTime) {
			t.Errorf("%s: ModTime = %v, want %v", hdr.Name, hdr.ModTime, meta.ModTime)
		}
		if hdr.Uid != meta.Uid || hdr.Gid != meta.Gid {
			t.Errorf("%s: uid/gid = %d/%d, want %d/%d", hdr.Name, hdr.Uid, hdr.Gid, meta.Uid, meta.Gid)
		}
		if hdr.Uname != meta.Uname || hdr.Gname != meta.Gname {
			t.Errorf("%s: uname/gname = %q/%q, want %q/%q", hdr.Name, hdr.Uname, hdr.Gname, meta.Uname, meta.Gname)
		}
		if hdr.AccessTime != (time.Time{}) || hdr.ChangeTime != (time.Time{}) {
			t.Errorf("%s: host atime/ctime leaked into header", hdr.Name)
		}
		if len(hdr.PAXRecords) > 0 {
			// PAX records may only be auto-generated path/size records, never host xattrs.
			for k := range hdr.PAXRecords {
				if strings.HasPrefix(k, "SCHILY.xattr") || strings.HasPrefix(k, "LIBARCHIVE.xattr") {
					t.Errorf("%s: host xattr leaked into PAX records: %s", hdr.Name, k)
				}
			}
		}

		switch hdr.Name {
		case "dir":
			if hdr.Typeflag != tar.TypeDir {
				t.Errorf("dir: Typeflag = %c, want TypeDir", hdr.Typeflag)
			}
			if hdr.Mode&0o777 != 0o755 {
				t.Errorf("dir: perms = %o, want 755", hdr.Mode)
			}
		case "root.txt":
			body, _ := io.ReadAll(tr)
			if string(body) != "root contents\n" {
				t.Errorf("root.txt content = %q", body)
			}
			if hdr.Mode&0o777 != 0o640 {
				t.Errorf("root.txt perms = %o, want 640", hdr.Mode)
			}
		case "link-to-root":
			if hdr.Typeflag != tar.TypeSymlink || hdr.Linkname != "root.txt" {
				t.Errorf("link-to-root: type=%c target=%q", hdr.Typeflag, hdr.Linkname)
			}
		case "文件名-日本語.txt":
			body, _ := io.ReadAll(tr)
			if string(body) != "unicode body" {
				t.Errorf("unicode file content = %q", body)
			}
		case "a-hard.txt":
			if hdr.Typeflag != tar.TypeReg {
				t.Errorf("anchor a-hard.txt Typeflag = %c, want TypeReg", hdr.Typeflag)
			}
			body, _ := io.ReadAll(tr)
			if string(body) != "shared hardlink body" {
				t.Errorf("anchor content = %q", body)
			}
		case "m-hard.txt", "z-hard.txt":
			if hdr.Typeflag != tar.TypeLink {
				t.Errorf("%s: Typeflag = %c, want TypeLink", hdr.Name, hdr.Typeflag)
			}
			if hdr.Linkname != "a-hard.txt" {
				t.Errorf("%s: Linkname = %q, want anchor a-hard.txt", hdr.Name, hdr.Linkname)
			}
			if hdr.Size != 0 {
				t.Errorf("%s: hard link size = %d, want 0", hdr.Name, hdr.Size)
			}
		}
	}

	expected := []string{
		"a-hard.txt", "dir", "dir/a.txt", "link-to-root", "m-hard.txt",
		"root.txt", "z-hard.txt", "zzz/last.txt", "文件名-日本語.txt",
	}
	if len(seen) != len(expected) {
		t.Fatalf("saw %d entries, want %d: %v", len(seen), len(expected), seen)
	}
	for _, name := range expected {
		if !seen[name] {
			t.Errorf("missing entry %q", name)
		}
	}
}

// ---- ZIP determinism and normalization -------------------------------------

func TestCanonicalZipDeterministic(t *testing.T) {
	for _, tpl := range []Zip{
		{Compression: kzip.Store},
		{SelectiveCompression: true, Compression: kzip.Deflate},
	} {
		t.Run(fmt.Sprintf("method=%d", tpl.Compression), func(t *testing.T) {
			ca := testCanonical()
			ca.Template = tpl
			ca.Hardlinks = CanonicalHardlinkMaterialize

			baseFiles := buildFiles(canonicalTree())
			reference := archiveBytes(t, ca, baseFiles)

			if got := archiveBytes(t, ca, buildFiles(canonicalTree())); !bytes.Equal(got, reference) {
				t.Error("repeated canonical zip archiving produced different bytes")
			}
			rng := rand.New(rand.NewPCG(7, 13))
			for i := 0; i < 8; i++ {
				got := archiveBytes(t, ca, shuffledCopy(rng, baseFiles))
				if !bytes.Equal(got, reference) {
					t.Fatalf("permutation %d changed canonical zip bytes", i)
				}
			}
			for vi, variant := range hostVariants(canonicalTree()) {
				got := archiveBytes(t, ca, shuffledCopy(rng, buildFiles(variant)))
				if !bytes.Equal(got, reference) {
					t.Fatalf("host variant %d changed canonical zip bytes", vi)
				}
			}

			verifyZipContents(t, reference, ca.Metadata)
		})
	}
}

func verifyZipContents(t *testing.T, data []byte, meta CanonicalMetadata) {
	t.Helper()
	zr, err := kzip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		t.Fatalf("zip.NewReader: %v", err)
	}

	seen := map[string]bool{}
	for _, f := range zr.File {
		seen[f.Name] = true

		if f.Comment != "" {
			t.Errorf("%s: non-empty comment %q", f.Name, f.Comment)
		}
		if !f.Modified.UTC().Equal(meta.ModTime.UTC()) {
			t.Errorf("%s: Modified = %v, want %v", f.Name, f.Modified.UTC(), meta.ModTime.UTC())
		}

		// Only deterministic, auto-generated extras are allowed.
		for _, id := range parseZipExtraIDs(t, f.Name, f.Extra) {
			switch id {
			case 0x5455, 0x0001: // extended timestamp, zip64
			default:
				t.Errorf("%s: unexpected extra field id 0x%04x", f.Name, id)
			}
		}

		// UTF-8 flag: clear for pure ASCII, set for non-ASCII.
		hasUTF8Flag := f.Flags&0x800 != 0
		if isASCII([]byte(strings.TrimSuffix(f.Name, "/"))) && hasUTF8Flag {
			t.Errorf("%s: UTF-8 flag set for ASCII name", f.Name)
		}
		if !isASCII([]byte(strings.TrimSuffix(f.Name, "/"))) && !hasUTF8Flag {
			t.Errorf("%s: UTF-8 flag missing for non-ASCII name", f.Name)
		}

		rc, err := f.Open()
		if err != nil {
			t.Fatalf("%s: open: %v", f.Name, err)
		}
		body, _ := io.ReadAll(rc)
		rc.Close()

		switch strings.TrimSuffix(f.Name, "/") {
		case "root.txt":
			if string(body) != "root contents\n" {
				t.Errorf("root.txt content = %q", body)
			}
			if f.Mode().Perm() != 0o640 {
				t.Errorf("root.txt perms = %o", f.Mode().Perm())
			}
		case "link-to-root":
			if f.Mode()&fs.ModeSymlink == 0 {
				t.Errorf("link-to-root lost symlink mode")
			}
			if string(body) != "root.txt" {
				t.Errorf("link-to-root body = %q, want link target", body)
			}
		case "a-hard.txt", "m-hard.txt", "z-hard.txt":
			if f.Mode()&fs.ModeSymlink != 0 {
				t.Errorf("%s: materialized hard link must be a regular file", f.Name)
			}
			if string(body) != "shared hardlink body" {
				t.Errorf("%s: materialized content = %q", f.Name, body)
			}
		case "文件名-日本語.txt":
			if string(body) != "unicode body" {
				t.Errorf("unicode content = %q", body)
			}
		}
	}

	for _, name := range []string{"dir/", "dir/a.txt", "root.txt", "link-to-root", "a-hard.txt", "m-hard.txt", "z-hard.txt", "zzz/last.txt", "文件名-日本語.txt"} {
		if !seen[name] {
			t.Errorf("missing zip entry %q", name)
		}
	}
}

func TestCanonicalZipHardlinkReject(t *testing.T) {
	ca := testCanonical()
	ca.Template = Zip{Compression: kzip.Store}
	ca.Hardlinks = CanonicalHardlinkReject

	err := ca.Archive(context.Background(), failOnWriteWriter{}, buildFiles(canonicalTree()))
	if err == nil {
		t.Fatal("expected rejection of hard links in zip, got nil error")
	}

	// Without hard links, rejection mode works fine.
	simple := []memEntry{
		{path: "a.txt", mode: 0o644, content: []byte("a")},
	}
	var buf bytes.Buffer
	if err := ca.Archive(context.Background(), &buf, buildFiles(simple)); err != nil {
		t.Fatalf("archive without hard links failed: %v", err)
	}
}

// ---- path validation and collisions ----------------------------------------

func TestCanonicalPathValidation(t *testing.T) {
	regular := func(p string) memEntry {
		return memEntry{path: p, mode: 0o644, content: []byte("x")}
	}

	badCases := []struct {
		name    string
		entries []memEntry
	}{
		{"empty path", []memEntry{{path: "", mode: 0o644, content: []byte("x")}}},
		{"dot path", []memEntry{regular(".")}},
		{"absolute", []memEntry{regular("/etc/passwd")}},
		{"parent escape", []memEntry{regular("../outside")}},
		{"cleaned parent escape", []memEntry{regular("a/../../outside")}},
		{"nul byte", []memEntry{regular("a\x00b")}},
		{"exact duplicate", []memEntry{regular("a/b"), regular("a/b")}},
		{"duplicate after clean", []memEntry{regular("a//b"), regular("a/./b")}},
		{"backslash normalized collision", []memEntry{regular("a\\b"), regular("a/b")}},
		{"dir and file same name", []memEntry{
			{path: "d", mode: fs.ModeDir | 0o755},
			{path: "d/", mode: 0o644, content: []byte("x")},
		}},
		{"file used as directory", []memEntry{regular("a"), regular("a/b")}},
	}

	for _, tc := range badCases {
		t.Run(tc.name, func(t *testing.T) {
			for _, tpl := range []any{Tar{}, Zip{}} {
				ca := testCanonical()
				ca.Template = tpl
				if err := ca.Archive(context.Background(), failOnWriteWriter{}, buildFiles(tc.entries)); err == nil {
					t.Errorf("%s (%T): expected error, got nil", tc.name, tpl)
				}
			}
		})
	}

	// A real directory entry sharing its prefix with child entries is valid.
	good := []memEntry{
		{path: "a", mode: fs.ModeDir | 0o755},
		{path: "a/b", mode: 0o644, content: []byte("b")},
	}
	ca := testCanonical()
	var buf bytes.Buffer
	if err := ca.Archive(context.Background(), &buf, buildFiles(good)); err != nil {
		t.Fatalf("directory-prefixed entries should be valid: %v", err)
	}
}

// ---- owner metadata ---------------------------------------------------------

func TestCanonicalZeroOwner(t *testing.T) {
	// Zero uid/gid must be written as zero, not treated as "leave host value".
	ca := testCanonical()
	ca.Metadata.Uid, ca.Metadata.Gid = 0, 0
	ca.Metadata.Uname, ca.Metadata.Gname = "", ""

	hostHdr := &tar.Header{Uid: 4242, Gid: 4343, Uname: "hostuser", PAXRecords: map[string]string{"SCHILY.xattr.user.foo": "bar"}}
	files := buildFiles([]memEntry{{
		path: "f.txt", mode: 0o644, content: []byte("f"), hostHeader: hostHdr,
	}})

	data := archiveBytes(t, ca, files)
	tr := tar.NewReader(bytes.NewReader(data))
	hdr, err := tr.Next()
	if err != nil {
		t.Fatalf("tar.Next: %v", err)
	}
	if hdr.Uid != 0 || hdr.Gid != 0 {
		t.Errorf("uid/gid = %d/%d, want 0/0", hdr.Uid, hdr.Gid)
	}
	if hdr.Uname != "" || hdr.Gname != "" {
		t.Errorf("uname/gname = %q/%q, want empty", hdr.Uname, hdr.Gname)
	}
	if _, leaked := hdr.PAXRecords["SCHILY.xattr.user.foo"]; leaked {
		t.Error("host PAX xattr record leaked into canonical header")
	}
}

func TestCanonicalNumericOwner(t *testing.T) {
	ca := testCanonical()
	ca.Metadata.NumericOwner = true
	data := archiveBytes(t, ca, buildFiles([]memEntry{
		{path: "f", mode: 0o644, content: []byte("x")},
	}))
	hdr, err := tar.NewReader(bytes.NewReader(data)).Next()
	if err != nil {
		t.Fatal(err)
	}
	if hdr.Uid != 1000 || hdr.Gid != 2000 {
		t.Errorf("uid/gid = %d/%d", hdr.Uid, hdr.Gid)
	}
	if hdr.Uname != "" || hdr.Gname != "" {
		t.Errorf("owner names not omitted: %q/%q", hdr.Uname, hdr.Gname)
	}
}

func TestCanonicalFixedMode(t *testing.T) {
	ca := testCanonical()
	ca.Metadata.ModePolicy = CanonicalModeFixed
	ca.Metadata.FixedMode = 0o640
	data := archiveBytes(t, ca, buildFiles([]memEntry{
		{path: "f", mode: 0o777, content: []byte("x")},
		{path: "d", mode: fs.ModeDir | 0o700},
	}))
	tr := tar.NewReader(bytes.NewReader(data))
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		if hdr.Mode&0o777 != 0o640 {
			t.Errorf("%s: perms = %o, want 640", hdr.Name, hdr.Mode)
		}
	}
}

// ---- hard-link anchor selection ---------------------------------------------

func TestCanonicalHardlinkAnchor(t *testing.T) {
	inode := &fakeInode{dev: 1, ino: 42}
	mkSet := func(extra string) []memEntry {
		entries := []memEntry{
			{path: "z/c", mode: 0o644, content: []byte("body"), inode: inode},
			{path: "a/b", mode: 0o644, content: []byte("body"), inode: inode},
			{path: "m/n", mode: 0o644, content: []byte("body"), inode: inode},
		}
		if extra != "" {
			entries = append(entries, memEntry{path: extra, mode: 0o644, content: []byte("body"), inode: inode})
		}
		return entries
	}

	ca := testCanonical()
	reference := archiveBytes(t, ca, buildFiles(mkSet("")))

	// Sorted anchor is "a/b"; every permutation must produce identical bytes.
	rng := rand.New(rand.NewPCG(3, 4))
	for i := 0; i < 5; i++ {
		got := archiveBytes(t, ca, shuffledCopy(rng, buildFiles(mkSet(""))))
		if !bytes.Equal(got, reference) {
			t.Fatalf("hard-link set permutation %d changed bytes", i)
		}
	}

	links := tarLinks(t, reference)
	if links["z/c"] != "a/b" || links["m/n"] != "a/b" {
		t.Fatalf("anchor resolution wrong: %v", links)
	}
	if _, isLink := links["a/b"]; isLink {
		t.Error("anchor must not itself be a hard link")
	}

	// Adding a smaller canonical path moves the anchor and repoints every link.
	moved := archiveBytes(t, ca, buildFiles(mkSet("a/a")))
	if bytes.Equal(moved, reference) {
		t.Error("anchor change must alter the archive bytes")
	}
	links = tarLinks(t, moved)
	if links["a/b"] != "a/a" || links["z/c"] != "a/a" || links["m/n"] != "a/a" {
		t.Fatalf("anchor did not move to a/a: %v", links)
	}
}

func tarLinks(t *testing.T, data []byte) map[string]string {
	t.Helper()
	links := map[string]string{}
	tr := tar.NewReader(bytes.NewReader(data))
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			return links
		}
		if err != nil {
			t.Fatal(err)
		}
		if hdr.Typeflag == tar.TypeLink {
			links[hdr.Name] = hdr.Linkname
		}
	}
}

// ---- long names and PAX record ordering ------------------------------------

func TestCanonicalLongNames(t *testing.T) {
	longName := "deep/" + strings.Repeat("very-long-directory-name/", 10) + "file.txt"
	if len(longName) < 150 {
		t.Fatalf("test name not long enough: %d", len(longName))
	}
	longTarget := strings.Repeat("some-long-target-component/", 8) + "target.txt"
	entries := []memEntry{
		{path: longName, mode: 0o644, content: []byte("long name body")},
		// Both name and target exceed USTAR limits, forcing one PAX header
		// carrying both "path" and "linkpath" records whose order we can check.
		{path: "lnk/" + strings.Repeat("long-link-name/", 7) + "link", mode: fs.ModeSymlink | 0o777, target: longTarget},
	}

	// TAR: long names force PAX extended headers; records within each header
	// must be key-sorted (the standard library emits them in sorted order).
	ca := testCanonical()
	tarData := archiveBytes(t, ca, buildFiles(entries))
	keyGroups := paxHeaderKeyGroups(t, tarData)
	if len(keyGroups) == 0 {
		t.Fatal("expected at least one PAX extended header for the long names")
	}
	multiRecord := false
	for gi, keys := range keyGroups {
		if len(keys) >= 2 {
			multiRecord = true
		}
		for i := 1; i < len(keys); i++ {
			if keys[i-1] > keys[i] {
				t.Fatalf("PAX header %d record keys not sorted: %v", gi, keys)
			}
		}
	}
	if !multiRecord {
		t.Fatalf("expected a PAX header with multiple records, got groups: %v", keyGroups)
	}
	// Round trip preserves the full names, targets and content.
	tr := tar.NewReader(bytes.NewReader(tarData))
	foundReg, foundLink := false, false
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		switch hdr.Typeflag {
		case tar.TypeReg:
			foundReg = true
			if hdr.Name != longName {
				t.Errorf("tar long name = %q, want %q", hdr.Name, longName)
			}
			body, _ := io.ReadAll(tr)
			if string(body) != "long name body" {
				t.Errorf("tar long-name body = %q", body)
			}
		case tar.TypeSymlink:
			foundLink = true
			if hdr.Linkname != longTarget {
				t.Errorf("tar long symlink target = %q, want %q", hdr.Linkname, longTarget)
			}
			if len(hdr.Linkname) < 100 {
				t.Errorf("tar long symlink target was truncated: %q", hdr.Linkname)
			}
		}
	}
	if !foundReg || !foundLink {
		t.Fatalf("missing entries: reg=%v link=%v", foundReg, foundLink)
	}

	// ZIP: same long names round trip through its own headers.
	ca.Template = Zip{Compression: kzip.Store}
	zipData := archiveBytes(t, ca, buildFiles(entries))
	zr, err := kzip.NewReader(bytes.NewReader(zipData), int64(len(zipData)))
	if err != nil {
		t.Fatal(err)
	}
	if zr.File[0].Name != longName {
		t.Errorf("zip long name = %q, want %q", zr.File[0].Name, longName)
	}
	rc, _ := zr.File[0].Open()
	body, _ := io.ReadAll(rc)
	rc.Close()
	if string(body) != "long name body" {
		t.Errorf("zip long-name body = %q", body)
	}
	for _, f := range zr.File {
		if f.Mode()&fs.ModeSymlink != 0 {
			rc, _ := f.Open()
			target, _ := io.ReadAll(rc)
			rc.Close()
			if string(target) != longTarget {
				t.Errorf("zip long symlink target = %q, want %q", target, longTarget)
			}
		}
	}
}

// paxHeaderKeyGroups scans raw 512-byte tar blocks for PAX extended headers
// and returns the record keys of each header in their on-wire order.
func paxHeaderKeyGroups(t *testing.T, data []byte) [][]string {
	t.Helper()
	var groups [][]string
	for off := 0; off+512 <= len(data); off += 512 {
		block := data[off : off+512]
		if block[156] != tar.TypeXHeader {
			continue
		}
		sizeField := strings.TrimRight(string(block[124:136]), "\x00 ")
		size, err := strconv.ParseInt(sizeField, 8, 64)
		if err != nil {
			t.Fatalf("bad PAX header size %q: %v", sizeField, err)
		}
		payloadEnd := off + 512 + int(size)
		if payloadEnd > len(data) {
			t.Fatal("PAX header extends beyond archive")
		}
		// size describes the record payload that immediately follows the 512-byte
		// PAX header block (stream padding to a 512 boundary is ignored).
		payload := string(data[off+512 : payloadEnd])
		var keys []string
		for len(payload) > 0 {
			sp := strings.IndexByte(payload, ' ')
			if sp < 0 {
				t.Fatal("malformed PAX record")
			}
			recLen, err := strconv.Atoi(payload[:sp])
			if err != nil || recLen <= 0 || recLen > len(payload) {
				t.Fatalf("bad PAX record length %q", payload[:sp])
			}
			rec := payload[:recLen]
			eq := strings.IndexByte(rec, '=')
			if eq < 0 {
				t.Fatal("PAX record missing '='")
			}
			keys = append(keys, rec[sp+1:eq])
			payload = payload[recLen:]
		}
		groups = append(groups, keys)
	}
	return groups
}

// ---- size mismatch and lazy opens ------------------------------------------

func TestCanonicalSizeMismatch(t *testing.T) {
	cases := []struct {
		name string
		e    memEntry
	}{
		{
			name: "short content",
			e: memEntry{
				path:          "short.txt",
				mode:          0o644,
				content:       []byte("full declared body"),
				actualContent: []byte("short"),
			},
		},
		{
			name: "content longer than declared",
			e: memEntry{
				path:          "long.txt",
				mode:          0o644,
				content:       []byte("decl"),
				declaredSize:  4,
				actualContent: []byte("declared body is much longer"),
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// A directory before and a regular file after the bad entry prove
			// that failure aborts the whole archive rather than continuing.
			entries := []memEntry{
				{path: "d", mode: fs.ModeDir | 0o755},
				tc.e,
				{path: "after.txt", mode: 0o644, content: []byte("after")},
			}
			for _, tpl := range []any{Tar{}, Zip{Compression: kzip.Store}} {
				ca := testCanonical()
				ca.Template = tpl
				var buf bytes.Buffer
				err := ca.Archive(context.Background(), &buf, buildFiles(entries))
				if err == nil {
					t.Errorf("%T: expected size-mismatch error, got nil", tpl)
				}
			}
		})
	}
}

func TestCanonicalLazyOpen(t *testing.T) {
	dirOpens, linkOpens, anchorOpens, memberOpens1, memberOpens2, regularOpens := 0, 0, 0, 0, 0, 0
	inode := &fakeInode{dev: 3, ino: 5}
	entries := []memEntry{
		{path: "d", mode: fs.ModeDir | 0o755, openCount: &dirOpens},
		{path: "sym", mode: fs.ModeSymlink | 0o777, target: "reg", openCount: &linkOpens},
		{path: "reg", mode: 0o644, content: []byte("reg"), openCount: &regularOpens},
		{path: "z-link", mode: 0o644, content: []byte("shared"), inode: inode, openCount: &memberOpens1},
		{path: "a-anchor", mode: 0o644, content: []byte("shared"), inode: inode, openCount: &anchorOpens},
		{path: "m-link", mode: 0o644, content: []byte("shared"), inode: inode, openCount: &memberOpens2},
	}

	// TAR: only the anchor's body is read; links, dirs and symlinks are not opened.
	ca := testCanonical()
	archiveBytes(t, ca, buildFiles(entries))
	if dirOpens != 0 || linkOpens != 0 {
		t.Errorf("tar: dir/symlink were opened: dir=%d sym=%d", dirOpens, linkOpens)
	}
	if anchorOpens != 1 || memberOpens1 != 0 || memberOpens2 != 0 {
		t.Errorf("tar: hard-link opens anchor=%d members=%d/%d, want 1/0/0", anchorOpens, memberOpens1, memberOpens2)
	}
	if regularOpens != 1 {
		t.Errorf("tar: regular file opened %d times, want 1", regularOpens)
	}

	// ZIP with materialization: every regular entry is opened exactly once.
	dirOpens, linkOpens, anchorOpens, memberOpens1, memberOpens2, regularOpens = 0, 0, 0, 0, 0, 0
	ca.Template = Zip{Compression: kzip.Store}
	ca.Hardlinks = CanonicalHardlinkMaterialize
	archiveBytes(t, ca, buildFiles(entries))
	if dirOpens != 0 || linkOpens != 0 {
		t.Errorf("zip: dir/symlink were opened: dir=%d sym=%d", dirOpens, linkOpens)
	}
	if anchorOpens != 1 || memberOpens1 != 1 || memberOpens2 != 1 || regularOpens != 1 {
		t.Errorf("zip: opens anchor=%d members=%d/%d regular=%d, all want 1", anchorOpens, memberOpens1, memberOpens2, regularOpens)
	}
}

// ---- input immutability -----------------------------------------------------

func TestCanonicalInputNotMutated(t *testing.T) {
	entries := canonicalTree()
	files := buildFiles(entries)

	snapshot := make([]FileInfo, len(files))
	for i, f := range files {
		snapshot[i] = f // shallow snapshot of the value the caller owns
	}

	ca := testCanonical()
	ca.Template = Zip{Compression: kzip.Deflate}
	ca.Hardlinks = CanonicalHardlinkMaterialize
	archiveBytes(t, ca, files)

	for i := range files {
		got, want := files[i], snapshot[i]
		if got.NameInArchive != want.NameInArchive || got.LinkTarget != want.LinkTarget {
			t.Errorf("file %d: archive path/link target mutated: %q/%q vs %q/%q",
				i, got.NameInArchive, got.LinkTarget, want.NameInArchive, want.LinkTarget)
		}
		if got.Mode() != want.Mode() {
			t.Errorf("file %d: mode mutated: %v vs %v", i, got.Mode(), want.Mode())
		}
		if !got.ModTime().Equal(want.ModTime()) {
			t.Errorf("file %d: modtime mutated: %v vs %v", i, got.ModTime(), want.ModTime())
		}
		if got.Sys() != want.Sys() {
			t.Errorf("file %d: Sys identity changed", i)
		}
		if got.Header != want.Header {
			t.Errorf("file %d: Header field mutated", i)
		}
		if got.Size() != want.Size() {
			t.Errorf("file %d: size mutated: %d vs %d", i, got.Size(), want.Size())
		}
	}
}

// ---- misc rejection and configuration --------------------------------------

func TestCanonicalUnsupportedEntryType(t *testing.T) {
	for _, mode := range []fs.FileMode{fs.ModeNamedPipe | 0o644, fs.ModeDevice | 0o644, fs.ModeSocket | 0o644} {
		ca := testCanonical()
		err := ca.Archive(context.Background(), failOnWriteWriter{}, buildFiles([]memEntry{
			{path: "weird", mode: mode},
		}))
		if err == nil {
			t.Errorf("mode %v: expected rejection", mode)
		}
	}
}

func TestCanonicalTemplateGuard(t *testing.T) {
	ca := testCanonical()
	ca.Template = "not-a-template"
	if err := ca.Archive(context.Background(), failOnWriteWriter{}, buildFiles([]memEntry{
		{path: "a", mode: 0o644, content: []byte("a")},
	})); err == nil {
		t.Error("expected error for unsupported template")
	}

	ca.Template = nil
	if err := ca.Archive(context.Background(), &bytes.Buffer{}, nil); err == nil {
		t.Error("expected error for nil template")
	}
}

func TestCanonicalZeroModTime(t *testing.T) {
	ca := testCanonical()
	ca.Metadata.ModTime = time.Time{}
	ca.Template = Zip{Compression: kzip.Store}
	data := archiveBytes(t, ca, buildFiles([]memEntry{
		{path: "a.txt", mode: 0o644, content: []byte("a")},
	}))
	zr, err := kzip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		t.Fatal(err)
	}
	f := zr.File[0]
	// An explicit zero policy must leave no timestamp on disk: both legacy
	// MS-DOS fields are zero and no extended-timestamp extra is emitted.
	if f.ModifiedDate != 0 || f.ModifiedTime != 0 {
		t.Errorf("zero ModTime produced dos fields %04x/%04x", f.ModifiedDate, f.ModifiedTime)
	}
	for _, id := range parseZipExtraIDs(t, f.Name, f.Extra) {
		if id == 0x5455 {
			t.Error("extended-timestamp extra must be absent for zero ModTime")
		}
	}
}

func TestCanonicalContextCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	ca := testCanonical()
	err := ca.Archive(ctx, &bytes.Buffer{}, buildFiles(canonicalTree()))
	if !errors.Is(err, context.Canceled) {
		t.Errorf("expected context.Canceled, got %v", err)
	}
}

func TestCanonicalInvalidUTF8RejectedForZip(t *testing.T) {
	ca := testCanonical()
	ca.Template = Zip{}
	bad := append([]byte("a/"), 0xff, 0xfe)
	files := []FileInfo{{
		FileInfo:      memFileInfo{name: "x", size: 1, mode: 0o644},
		NameInArchive: string(bad),
		Open:          func() (fs.File, error) { return &memFile{Reader: bytes.NewReader([]byte("x"))}, nil },
	}}
	if err := ca.Archive(context.Background(), failOnWriteWriter{}, files); err == nil {
		t.Error("expected invalid UTF-8 path to be rejected for zip")
	}
}

// Ensure the regular os.FileInfo-based path compiles and behaves for nil
// Sys values (common for in-memory fs.FS implementations).
func TestCanonicalNilSysAccepted(t *testing.T) {
	ca := testCanonical()
	ca.hardlinkKeyFunc = nil // fall back to the platform default
	files := []FileInfo{{
		FileInfo:      memFileInfo{name: "a.txt", size: 1, mode: 0o644, modTime: time.Now(), sys: nil},
		NameInArchive: "a.txt",
		Open:          func() (fs.File, error) { return &memFile{Reader: bytes.NewReader([]byte("x"))}, nil },
	}}
	var buf bytes.Buffer
	if err := ca.Archive(context.Background(), &buf, files); err != nil {
		t.Fatalf("nil Sys should be accepted: %v", err)
	}
	if buf.Len() == 0 {
		t.Error("archive is empty")
	}
}

// parseZipExtraIDs parses raw zip extra-field bytes into their 16-bit IDs.
func parseZipExtraIDs(t *testing.T, name string, extra []byte) []uint16 {
	t.Helper()
	var ids []uint16
	for len(extra) >= 4 {
		id := binary.LittleEndian.Uint16(extra[0:2])
		size := int(binary.LittleEndian.Uint16(extra[2:4]))
		if 4+size > len(extra) {
			t.Fatalf("%s: truncated extra field", name)
		}
		ids = append(ids, id)
		extra = extra[4+size:]
	}
	if len(extra) != 0 {
		t.Fatalf("%s: trailing %d bytes in extra field", name, len(extra))
	}
	return ids
}
