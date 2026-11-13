//go:build !windows

package archives

import (
	"archive/tar"
	"bytes"
	"context"
	"io"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// TestCanonicalTarRealHardlinks exercises the full disk-based pipeline:
// FilesFromDisk discovers hard links in walk order (which is not under our
// control), and CanonicalArchive must still choose the lexicographically
// smallest canonical path as anchor and produce identical bytes on every
// run regardless of map iteration order.
func TestCanonicalTarRealHardlinks(t *testing.T) {
	tmp := t.TempDir()

	content := []byte("real inode content\n")
	// Deliberately name members so the anchor ("aaa") is not the first
	// entry a directory walk is guaranteed to visit.
	paths := map[string]string{
		"zzz.txt": filepath.Join(tmp, "zzz.txt"),
		"mmm.txt": filepath.Join(tmp, "mmm.txt"),
		"aaa.txt": filepath.Join(tmp, "aaa.txt"),
	}
	if err := os.WriteFile(paths["zzz.txt"], content, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Link(paths["zzz.txt"], paths["mmm.txt"]); err != nil {
		t.Fatal(err)
	}
	if err := os.Link(paths["zzz.txt"], paths["aaa.txt"]); err != nil {
		t.Fatal(err)
	}

	ca := CanonicalArchive{
		Template: Tar{},
		Metadata: CanonicalMetadata{
			ModTime: time.Date(2021, 6, 7, 8, 9, 10, 0, time.UTC),
			Uid:     0,
			Gid:     0,
		},
	}

	ctx := context.Background()
	var first []byte
	for run := 0; run < 3; run++ {
		files, err := FilesFromDisk(ctx, nil, map[string]string{tmp + string(filepath.Separator): ""})
		if err != nil {
			t.Fatalf("FilesFromDisk: %v", err)
		}

		var buf bytes.Buffer
		if err := ca.Archive(ctx, &buf, files); err != nil {
			t.Fatalf("canonical Archive run %d: %v", run, err)
		}
		if first == nil {
			first = buf.Bytes()
		} else if !bytes.Equal(buf.Bytes(), first) {
			t.Fatalf("run %d produced different bytes", run)
		}
	}

	// Independently unpack and verify the anchor relationship and contents.
	tr := tar.NewReader(bytes.NewReader(first))
	anchorSeen := false
	links := map[string]string{}
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
			if hdr.Name != "aaa.txt" {
				t.Errorf("unexpected regular entry %q, want anchor aaa.txt", hdr.Name)
			}
			body, _ := io.ReadAll(tr)
			if !bytes.Equal(body, content) {
				t.Errorf("anchor content = %q, want %q", body, content)
			}
			anchorSeen = true
		case tar.TypeLink:
			links[hdr.Name] = hdr.Linkname
		}
	}
	if !anchorSeen {
		t.Fatal("anchor aaa.txt was not written as a regular file")
	}
	if len(links) != 2 || links["mmm.txt"] != "aaa.txt" || links["zzz.txt"] != "aaa.txt" {
		t.Fatalf("hard-link resolution wrong: %v", links)
	}
}
