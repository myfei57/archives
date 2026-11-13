package archives

import (
	"archive/tar"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"hash"
	"io"
	"io/fs"
	"path"
	"slices"
	"strings"
	"time"
)

// This file implements a one-pass, random-access index for uncompressed tar
// archives. BuildTarIndex scans an immutable io.ReaderAt exactly once and
// records the physical header/body ranges of every entry. IndexedTarFS then
// serves Open/ReadAt/ReadDir/Sub requests straight from those ranges without
// ever re-scanning the archive.
//
// Design invariants:
//
//   - The serialized index contains data only: no executable callbacks, no
//     open functions, and no host file system paths.
//   - Source identity is a full-content digest (never size/mtime, which are
//     not content identities) plus the source length, bound together with an
//     index format version.
//   - A decoded index is structurally validated (bounds, alignment, overlap,
//     duplicate names, hardlink graph) before any source range is accessed.
//   - Sparse archives are rejected explicitly: a contiguous SectionReader
//     cannot represent hole/data extents, so we never silently return the
//     raw on-disk bytes as if they were logical content.
//   - The caller owns the source; IndexedTarFS never closes it.

const (
	// tarIndexVersion is the current on-disk/index format version. It is
	// bumped whenever the serialized TarIndex contract changes.
	tarIndexVersion = 1

	// tarIndexDigestAlgo is the digest algorithm bound to every index.
	tarIndexDigestAlgo = "sha256"

	// tarIndexBlockSize is the tar block size; every header and padded body
	// is aligned to this boundary.
	tarIndexBlockSize = 512

	// maxInt64 avoids pulling in math just for one boundary check.
	maxInt64 = int64(^uint64(0) >> 1)
)

// Sentinel errors returned by the index builder and file system. Callers
// should use errors.Is rather than matching message text.
var (
	// ErrSparseTarUnsupported is returned when an archive uses sparse tar
	// features (GNU/PAX sparse maps). Data extent mapping is not implemented,
	// so serving logical content from a contiguous byte range would expose
	// hole data incorrectly; the archive is rejected instead.
	ErrSparseTarUnsupported = errors.New("sparse tar archives are not supported by the tar index")

	// ErrCorruptTarIndex is returned when an index fails structural
	// validation (bad version, out-of-bounds or overlapping ranges,
	// duplicate names, broken hardlink graph, etc.). The error is returned
	// before any byte of the indexed source is read.
	ErrCorruptTarIndex = errors.New("corrupt tar index")

	// ErrUnsupportedTarEntry is returned when opening an indexed entry whose
	// type cannot be represented by an fs.FS (character/block devices, FIFOs).
	ErrUnsupportedTarEntry = errors.New("unsupported tar entry type")

	// ErrHardlinkCycle is returned when hardlink references form a cycle.
	ErrHardlinkCycle = errors.New("hardlink cycle detected")

	// ErrHardlinkTarget is returned when a hardlink points at a missing or
	// non-regular (non-file) entry.
	ErrHardlinkTarget = errors.New("invalid hardlink target")
)

// TarDuplicatePolicy decides what the index builder does when several archive
// entries normalize to the same logical name.
type TarDuplicatePolicy int

const (
	// TarDuplicateError makes the build fail on duplicate logical names.
	// Use this for strict validation of author-controlled archives.
	TarDuplicateError TarDuplicatePolicy = iota

	// TarDuplicateKeepFirst keeps the first entry seen in archive order and
	// discards later entries with the same logical name.
	TarDuplicateKeepFirst

	// TarDuplicateKeepLast keeps the last entry seen in archive order and
	// discards earlier entries with the same logical name, roughly matching
	// the behavior of extracting over an existing tree. This is the default
	// because real-world tars legitimately repeat directory entries (for
	// example when archives are appended to).
	TarDuplicateKeepLast
)

// TarIndex is a serializable, content-addressed index of an uncompressed tar
// archive. All exported fields are plain data and may be encoded with
// encoding/json (see EncodeTarIndex/DecodeTarIndex). The index stores no
// callbacks, open functions, or host file system paths; runtime lookup tables
// are unexported and rebuilt after decoding.
type TarIndex struct {
	// Version is the index format version (tarIndexVersion).
	Version int `json:"version"`

	// DigestAlgo is the name of the digest algorithm used for Digest.
	DigestAlgo string `json:"digest_algo"`

	// Digest is the full-content digest of the indexed source bytes,
	// computed during the single indexing scan. Size and modification time
	// are deliberately never used as content identity.
	Digest []byte `json:"digest"`

	// Size is the exact length in bytes of the indexed source. Every
	// indexed range is validated to lie within [0, Size).
	Size int64 `json:"size"`

	// Entries are the indexed entries in archive order (meta records such as
	// PAX and GNU long-name headers are consumed by the parser and are not
	// entries).
	Entries []TarEntry `json:"entries"`

	// byName maps cleaned logical names to their entry. Rebuilt after a
	// decode; not part of the serialized contract.
	byName map[string]*TarEntry

	// dirs maps each directory name (including the implicit root ".") to
	// its sorted directory entries, with implicit parent directories filled
	// in. Rebuilt after a decode; not part of the serialized contract.
	dirs map[string][]fs.DirEntry
}

// TarEntry describes one indexed tar entry and the physical source ranges
// that hold its header chain and body.
type TarEntry struct {
	// Name is the cleaned, fs.ValidPath-compliant logical name.
	Name string `json:"name"`

	// Typeflag is the normalized tar type flag (tar.TypeReg, tar.TypeDir,
	// tar.TypeSymlink, tar.TypeLink, tar.TypeChar, tar.TypeBlock,
	// tar.TypeFifo, or tar.TypeCont treated as a regular file).
	Typeflag byte `json:"typeflag"`

	// Mode is the portable file mode (io/fs.FileMode bits), including type
	// bits such as fs.ModeDir and fs.ModeSymlink.
	Mode uint32 `json:"mode"`

	Uid   int    `json:"uid"`
	Gid   int    `json:"gid"`
	Uname string `json:"uname,omitempty"`
	Gname string `json:"gname,omitempty"`

	// ModTime is the header modification time.
	ModTime time.Time `json:"modtime"`

	// Size is the logical body length. For hardlinks it is copied from the
	// resolved target entry, since hardlinks share the target's content.
	Size int64 `json:"size"`

	// HeaderOff is the source offset where this entry's header chain starts.
	// The chain covers any local PAX and GNU long-name/long-link extension
	// blocks plus the final 512-byte header block immediately preceding the
	// body. Global PAX headers are shared state and are not attributed to any
	// single entry.
	HeaderOff int64 `json:"header_off"`

	// BodyOff is the source offset where the entry body starts. It is always
	// 512-byte aligned; the final header block occupies BodyOff-512.
	BodyOff int64 `json:"body_off"`

	// Linkname is the raw link target as recorded in the header (for
	// symlinks) or the cleaned target name (for hardlinks).
	Linkname string `json:"linkname,omitempty"`

	// Target is the resolved logical name of a hardlink's terminal regular
	// file. Empty for every other entry type. Forward references are
	// resolved after the whole archive has been indexed.
	Target string `json:"target,omitempty"`

	// TargetSize is the logical size shared with the hardlink target, since
	// hardlink headers themselves carry no body. Zero for other entry types.
	// Size always stays the physical body length used for range validation.
	TargetSize int64 `json:"target_size,omitempty"`

	// Format is the tar.Format guess reported by archive/tar for this entry.
	Format int `json:"format"`
}

// HeaderLen returns the total length of the header chain, including any PAX
// or GNU extension blocks and the final header block.
func (e TarEntry) HeaderLen() int64 { return e.BodyOff - e.HeaderOff }

// BodyEnd returns the exclusive end offset of the body.
func (e TarEntry) BodyEnd() int64 { return e.BodyOff + e.Size }

// TarIndexOption customizes index building.
type TarIndexOption func(*tarIndexConfig)

type tarIndexConfig struct {
	duplicatePolicy TarDuplicatePolicy
}

// WithTarDuplicatePolicy sets the strategy used when multiple entries
// normalize to the same logical name. The default is TarDuplicateKeepLast.
func WithTarDuplicatePolicy(policy TarDuplicatePolicy) TarIndexOption {
	return func(cfg *tarIndexConfig) { cfg.duplicatePolicy = policy }
}

// BuildTarIndex performs a single sequential scan of the uncompressed tar
// archive readable from src (whose exact length is size) and returns a
// TarIndex. PAX global/local extensions, GNU long names and long links,
// 512-byte body padding, and implicit directories are all handled. The full
// source digest is computed during the same pass, so the archive is read
// start to finish exactly once.
//
// The source must be immutable for the lifetime of the index: random reads
// later assume the indexed ranges still hold the same bytes.
func BuildTarIndex(ctx context.Context, src io.ReaderAt, size int64, opts ...TarIndexOption) (*TarIndex, error) {
	if src == nil {
		return nil, errors.New("tar index: nil source")
	}
	if size < 0 {
		return nil, fmt.Errorf("%w: negative source size %d", ErrCorruptTarIndex, size)
	}
	// Reserve headroom so block-alignment arithmetic can never overflow.
	if size > maxInt64-tarIndexBlockSize {
		return nil, fmt.Errorf("%w: source size %d too large", ErrCorruptTarIndex, size)
	}

	cfg := tarIndexConfig{duplicatePolicy: TarDuplicateKeepLast}
	for _, opt := range opts {
		opt(&cfg)
	}

	scan := &tarScanReader{r: io.NewSectionReader(src, 0, size), h: sha256.New()}
	tr := tar.NewReader(scan)

	idx := &TarIndex{
		Version:    tarIndexVersion,
		DigestAlgo: tarIndexDigestAlgo,
		Size:       size,
		byName:     make(map[string]*TarEntry),
	}

	for {
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		chainStart := scan.pos
		hdr, err := tr.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("tar index: scanning header at offset %d: %w", chainStart, err)
		}

		// Global PAX headers are returned as synthetic entries by
		// archive/tar after their body has been parsed and applied. They
		// describe subsequent entries, not files of their own.
		if hdr.Typeflag == tar.TypeXGlobalHeader {
			continue
		}

		// chainStart may point at the previous record's padding (for
		// example after a global PAX header whose body is not block-aligned).
		// The entry's own header chain begins at the next 512-byte boundary,
		// which is always exactly where archive/tar started reading its
		// extension and final header blocks.
		headerOff := (chainStart + tarIndexBlockSize - 1) / tarIndexBlockSize * tarIndexBlockSize

		// Reject sparse archives up front, before touching the body. A
		// contiguous SectionReader cannot reconstruct logical sparse
		// content (hole extents would be misreported as stored bytes).
		if tarHeaderIsSparse(hdr) {
			return nil, fmt.Errorf("tar index: entry %q: %w", hdr.Name, ErrSparseTarUnsupported)
		}

		bodyOff := scan.pos
		if bodyOff%tarIndexBlockSize != 0 {
			return nil, fmt.Errorf("%w: unaligned body offset %d for %q", ErrCorruptTarIndex, bodyOff, hdr.Name)
		}

		// Drain the logical body through archive/tar so its framing and
		// checksums are exercised and every source byte feeds the digest.
		n, err := io.Copy(io.Discard, tr)
		if err != nil {
			return nil, fmt.Errorf("tar index: reading body of %q: %w", hdr.Name, err)
		}
		bodyEnd := scan.pos
		if n != hdr.Size || bodyEnd != bodyOff+hdr.Size {
			return nil, fmt.Errorf("%w: size framing mismatch for %q", ErrCorruptTarIndex, hdr.Name)
		}
		if bodyEnd > size {
			return nil, fmt.Errorf("%w: body of %q extends past source end", ErrCorruptTarIndex, hdr.Name)
		}

		entry, err := buildTarEntry(hdr, headerOff, bodyOff)
		if err != nil {
			return nil, err
		}

		if err := idx.insertEntry(entry, cfg.duplicatePolicy); err != nil {
			return nil, err
		}
	}

	idx.Digest = scan.h.Sum(nil)

	if err := idx.finalize(); err != nil {
		return nil, err
	}
	return idx, nil
}

// tarScanReader is a sequential reader that tracks the current absolute
// offset and feeds every byte into a digest while archive/tar consumes it.
type tarScanReader struct {
	r   io.Reader
	pos int64
	h   hash.Hash
}

func (s *tarScanReader) Read(p []byte) (int, error) {
	n, err := s.r.Read(p)
	if n > 0 {
		s.pos += int64(n)
		s.h.Write(p[:n])
	}
	return n, err
}

// tarHeaderIsSparse reports whether hdr describes a sparse file. Go's
// archive/tar represents sparse files either with the GNU sparse type flag
// or with GNU.sparse.* PAX records; the public sparse fields differ across
// Go versions, so both canonical signals are checked here.
func tarHeaderIsSparse(hdr *tar.Header) bool {
	if hdr.Typeflag == tar.TypeGNUSparse {
		return true
	}
	for key := range hdr.PAXRecords {
		if strings.HasPrefix(key, "GNU.sparse.") {
			return true
		}
	}
	return false
}

// buildTarEntry converts a parsed tar header into an index entry, validating
// its type and normalized logical name.
func buildTarEntry(hdr *tar.Header, headerOff, bodyOff int64) (*TarEntry, error) {
	typeflag := hdr.Typeflag
	switch typeflag {
	case tar.TypeReg, tar.TypeDir, tar.TypeSymlink, tar.TypeLink,
		tar.TypeChar, tar.TypeBlock, tar.TypeFifo:
		// supported below
	case tar.TypeCont:
		// GNU contiguous files carry an ordinary body; treat as regular.
		typeflag = tar.TypeReg
	default:
		return nil, fmt.Errorf("%w: type flag %d (%q) for %q", ErrUnsupportedTarEntry, hdr.Typeflag, string(rune(hdr.Typeflag)), hdr.Name)
	}

	name := path.Clean(hdr.Name)
	// The root itself cannot be an indexed entry; it is always represented
	// by the implicit root directory. Entries literally named "." are
	// skipped rather than failing the build.
	if name == "." {
		return nil, nil
	}
	if !fs.ValidPath(name) {
		return nil, fmt.Errorf("%w: invalid entry path %q", ErrCorruptTarIndex, hdr.Name)
	}

	entry := &TarEntry{
		Name:      name,
		Typeflag:  typeflag,
		Mode:      uint32(hdr.FileInfo().Mode()),
		Uid:       hdr.Uid,
		Gid:       hdr.Gid,
		Uname:     hdr.Uname,
		Gname:     hdr.Gname,
		ModTime:   hdr.ModTime,
		Size:      hdr.Size,
		HeaderOff: headerOff,
		BodyOff:   bodyOff,
		Format:    int(hdr.Format),
	}

	switch typeflag {
	case tar.TypeSymlink:
		entry.Linkname = hdr.Linkname
	case tar.TypeLink:
		target := path.Clean(hdr.Linkname)
		if !fs.ValidPath(target) {
			return nil, fmt.Errorf("%w: hardlink %q has invalid target %q", ErrCorruptTarIndex, name, hdr.Linkname)
		}
		entry.Linkname = target
	}

	return entry, nil
}

// insertEntry applies the duplicate-name policy, including collisions produced
// by path normalization (e.g. "a//b" and "a/./b" both clean to "a/b").
func (idx *TarIndex) insertEntry(entry *TarEntry, policy TarDuplicatePolicy) error {
	if entry == nil {
		return nil
	}
	if existing, exists := idx.byName[entry.Name]; exists {
		switch policy {
		case TarDuplicateKeepFirst:
			return nil
		case TarDuplicateKeepLast:
			// Drop the earlier slot and append the new entry so the surviving
			// ranges stay in archive (monotonic) order. The byName map is only
			// used for presence checks during the scan; it is rebuilt from
			// Entries during finalize.
			for i := range idx.Entries {
				if &idx.Entries[i] == existing {
					idx.Entries = append(idx.Entries[:i], idx.Entries[i+1:]...)
					break
				}
			}
		default:
			return fmt.Errorf("%w: duplicate logical name %q", ErrCorruptTarIndex, entry.Name)
		}
	}
	idx.Entries = append(idx.Entries, *entry)
	idx.byName[entry.Name] = &idx.Entries[len(idx.Entries)-1]
	return nil
}

// finalize validates the complete index and builds the runtime lookup tables:
// name map, implicit directory tree, and the resolved hardlink graph.
func (idx *TarIndex) finalize() error {
	if idx.Version != tarIndexVersion {
		return fmt.Errorf("%w: unsupported version %d (want %d)", ErrCorruptTarIndex, idx.Version, tarIndexVersion)
	}
	if idx.DigestAlgo != tarIndexDigestAlgo || len(idx.Digest) != sha256.Size {
		return fmt.Errorf("%w: missing or unsupported source digest", ErrCorruptTarIndex)
	}
	if idx.Size < 0 {
		return fmt.Errorf("%w: negative source size", ErrCorruptTarIndex)
	}
	if idx.Size > maxInt64-tarIndexBlockSize {
		return fmt.Errorf("%w: source size %d too large", ErrCorruptTarIndex, idx.Size)
	}
	// Always rebuild the lookup tables from Entries: the scan-time map may
	// hold stale slots if a duplicate policy dropped entries.
	idx.byName = make(map[string]*TarEntry, len(idx.Entries))

	// Structural range validation: alignment, overflow, bounds, ordering and
	// overlap. Gaps between entries are legal (global PAX headers, end of
	// archive padding); overlaps and backwards motion are not.
	var prevEnd int64
	for i := range idx.Entries {
		e := &idx.Entries[i]
		if !fs.ValidPath(e.Name) {
			return fmt.Errorf("%w: invalid entry path %q", ErrCorruptTarIndex, e.Name)
		}
		if e.HeaderOff < 0 || e.BodyOff < 0 || e.Size < 0 {
			return fmt.Errorf("%w: negative offset/size for %q", ErrCorruptTarIndex, e.Name)
		}
		if e.HeaderOff%tarIndexBlockSize != 0 || e.BodyOff%tarIndexBlockSize != 0 {
			return fmt.Errorf("%w: unaligned range for %q", ErrCorruptTarIndex, e.Name)
		}
		headerLen := e.BodyOff - e.HeaderOff
		if headerLen < tarIndexBlockSize || headerLen%tarIndexBlockSize != 0 {
			return fmt.Errorf("%w: bad header range for %q", ErrCorruptTarIndex, e.Name)
		}
		// Overflow-checked body end.
		bodyEnd := e.BodyOff + e.Size
		if bodyEnd < e.BodyOff || bodyEnd > idx.Size {
			return fmt.Errorf("%w: body of %q is out of source bounds", ErrCorruptTarIndex, e.Name)
		}
		if e.HeaderOff < prevEnd {
			return fmt.Errorf("%w: overlapping or out-of-order range for %q", ErrCorruptTarIndex, e.Name)
		}
		prevEnd = bodyEnd

		if _, dup := idx.byName[e.Name]; dup {
			return fmt.Errorf("%w: duplicate logical name %q", ErrCorruptTarIndex, e.Name)
		}
		idx.byName[e.Name] = e
	}

	if err := idx.resolveHardlinks(); err != nil {
		return err
	}
	idx.buildDirs()
	return nil
}

// resolveHardlinks resolves every hardlink to its terminal regular file.
// Because the full index is available, forward references resolve naturally;
// dangling links, links to non-files, and reference cycles are rejected.
func (idx *TarIndex) resolveHardlinks() error {
	for i := range idx.Entries {
		e := &idx.Entries[i]
		if e.Typeflag != tar.TypeLink {
			continue
		}
		target, err := idx.followHardlink(e.Name)
		if err != nil {
			return err
		}
		e.Target = target.Name
		// Hardlinks share the target's content and therefore its logical
		// size; the physical Size is deliberately left untouched so range
		// validation keeps using the real (empty) body.
		if e.TargetSize != 0 && e.TargetSize != target.Size {
			return fmt.Errorf("%w: hardlink %q target size mismatch", ErrCorruptTarIndex, e.Name)
		}
		e.TargetSize = target.Size
	}
	return nil
}

// followHardlink walks the hardlink graph starting at name and returns the
// terminal regular entry, detecting cycles and invalid targets.
func (idx *TarIndex) followHardlink(name string) (*TarEntry, error) {
	visited := make(map[string]bool)
	current := name
	for {
		if visited[current] {
			return nil, fmt.Errorf("%w: %s", ErrHardlinkCycle, name)
		}
		visited[current] = true

		target, ok := idx.byName[current]
		if !ok {
			return nil, fmt.Errorf("%w: %q points at missing %q", ErrHardlinkTarget, name, current)
		}
		switch target.Typeflag {
		case tar.TypeLink:
			current = target.Linkname
		case tar.TypeReg:
			return target, nil
		default:
			return nil, fmt.Errorf("%w: %q points at non-regular %q", ErrHardlinkTarget, name, current)
		}
	}
}

// buildDirs constructs sorted per-directory listings, synthesizing implicit
// directories for parents that have no explicit directory entry.
func (idx *TarIndex) buildDirs() {
	idx.dirs = make(map[string][]fs.DirEntry)
	idx.dirs["."] = nil

	for i := range idx.Entries {
		e := &idx.Entries[i]
		dir := path.Dir(e.Name)
		idx.upsertDirEntry(dir, indexedTarEntryInfo{entry: e})

		// Walk the parent chain, inserting implicit directory entries until
		// reaching the implicit root.
		for parent := dir; parent != "."; parent = path.Dir(parent) {
			grandParent := path.Dir(parent)
			base := path.Base(parent)
			if !dirHasEntry(idx.dirs[grandParent], base) {
				idx.upsertDirEntry(grandParent, implicitDirInfo{implicitDirEntry{name: base}})
			}
		}
	}
}

// upsertDirEntry inserts info into dir's listing in name order, replacing an
// earlier implicit placeholder for the same name with the real entry.
func (idx *TarIndex) upsertDirEntry(dir string, info fs.DirEntry) {
	entries := idx.dirs[dir]
	position, found := slices.BinarySearchFunc(entries, info, func(a, b fs.DirEntry) int {
		return strings.Compare(a.Name(), b.Name())
	})
	if found {
		entries[position] = info
	} else {
		entries = slices.Insert(entries, position, info)
	}
	idx.dirs[dir] = entries
}

func dirHasEntry(entries []fs.DirEntry, name string) bool {
	for _, e := range entries {
		if e.Name() == name {
			return true
		}
	}
	return false
}

// EncodeTarIndex serializes an index to self-contained JSON data. The output
// holds plain data only (no callbacks, no host paths) and can be persisted or
// transferred independently of the archive bytes.
func EncodeTarIndex(idx *TarIndex) ([]byte, error) {
	if idx == nil {
		return nil, errors.New("tar index: cannot encode nil index")
	}
	return json.Marshal(exportTarIndex(idx))
}

// DecodeTarIndex parses serialized index data and performs full structural
// validation (including hardlink graph reconstruction) before returning. A
// corrupt index is rejected here, before any indexed source can be touched.
func DecodeTarIndex(data []byte) (*TarIndex, error) {
	var exported TarIndex
	if err := json.Unmarshal(data, &exported); err != nil {
		return nil, fmt.Errorf("%w: decoding: %v", ErrCorruptTarIndex, err)
	}
	exported.byName = nil
	exported.dirs = nil
	if err := exported.finalize(); err != nil {
		return nil, err
	}
	return &exported, nil
}

// exportTarIndex returns a value with the runtime lookup tables cleared so
// that only the serializable data contract is encoded.
func exportTarIndex(idx *TarIndex) TarIndex {
	copy := *idx
	copy.byName = nil
	copy.dirs = nil
	return copy
}

// IndexedTarFSOption customizes an indexed tar file system.
type IndexedTarFSOption func(*indexedFSConfig)

type indexedFSConfig struct {
	verifyDigest bool
}

// WithVerifyDigest requests explicit verification of the complete source
// digest when the file system is created. The whole source is hashed once
// and compared against the digest bound to the index; creation fails on any
// mismatch. Without this option only the index's structural validity and the
// source length are checked, and no full-archive read occurs.
func WithVerifyDigest(verify bool) IndexedTarFSOption {
	return func(cfg *indexedFSConfig) { cfg.verifyDigest = verify }
}

// IndexedTarFS is a read-only fs.FS (and fs.ReadDirFS, fs.StatFS, fs.SubFS,
// fs.ReadLinkFS, fs.ReadFileFS) backed by an immutable uncompressed tar
// source and a prebuilt TarIndex. After the index has been built, Open,
// ReadAt, ReadDir, Sub and Stat never rescan the archive: each file read
// touches only the indexed body range.
//
// The source is owned by the caller. Close is a no-op and will never close
// the underlying io.ReaderAt.
type IndexedTarFS struct {
	source  io.ReaderAt
	index   *TarIndex
	prefix  string // archive-relative subtree root, set by Sub
	rootErr error  // sticky construction-time error; Sub preserves it
}

// NewIndexedTarFS validates idx and binds it to src. Structural validation
// happens before any byte of src is read. With WithVerifyDigest, the full
// source digest is additionally verified before the file system is returned.
func NewIndexedTarFS(src io.ReaderAt, idx *TarIndex, opts ...IndexedTarFSOption) (*IndexedTarFS, error) {
	if src == nil {
		return nil, errors.New("tar index: nil source")
	}
	if idx == nil {
		return nil, fmt.Errorf("%w: nil index", ErrCorruptTarIndex)
	}
	cfg := indexedFSConfig{}
	for _, opt := range opts {
		opt(&cfg)
	}

	// Validate the index in isolation first; a corrupt index must never
	// cause a source range to be read. finalize is idempotent and also makes
	// sure the runtime lookup tables exist on the bound index.
	if err := idx.finalize(); err != nil {
		return nil, err
	}

	fsys := &IndexedTarFS{source: src, index: idx}

	if cfg.verifyDigest {
		h := sha256.New()
		if _, err := io.Copy(h, io.NewSectionReader(src, 0, idx.Size)); err != nil {
			return nil, fmt.Errorf("tar index: digest verification read failed: %w", err)
		}
		actual := h.Sum(nil)
		if !bytes.Equal(actual, idx.Digest) {
			return nil, fmt.Errorf("%w: source digest mismatch: index was built for different content", ErrCorruptTarIndex)
		}
	}
	return fsys, nil
}

// Open opens the named file. Directories return an fs.ReadDirFile; regular
// files (including hardlinks) return an fs.File that also implements
// io.ReaderAt and io.Seeker; symlinks are resolved to their in-archive
// target with loop protection. Each call returns an independent file handle
// with its own read offset.
func (f *IndexedTarFS) Open(name string) (fs.File, error) {
	if err := f.checkErr(); err != nil {
		return nil, err
	}
	if !fs.ValidPath(name) {
		return nil, &fs.PathError{Op: "open", Path: name, Err: fs.ErrInvalid}
	}
	archiveName := path.Join(f.prefix, name)

	if archiveName == "." {
		return f.openRoot()
	}

	entry, ok := f.index.byName[archiveName]
	if !ok {
		// Implicit directories are still openable.
		if _, isDir := f.index.dirs[archiveName]; isDir {
			return newIndexedDirFile(implicitDirInfo{implicitDirEntry{name: path.Base(archiveName)}}, f.index.dirs[archiveName]), nil
		}
		return nil, &fs.PathError{Op: "open", Path: name, Err: fs.ErrNotExist}
	}
	return f.openEntry(name, entry)
}

// openEntry materializes an fs.File for one indexed entry, following
// symlinks (but not hardlinks, which are already resolved to data).
func (f *IndexedTarFS) openEntry(requestedName string, entry *TarEntry) (fs.File, error) {
	switch entry.Typeflag {
	case tar.TypeDir:
		return newIndexedDirFile(indexedTarEntryInfo{entry: entry}, f.index.dirs[entry.Name]), nil

	case tar.TypeReg:
		return f.openRegular(entry), nil

	case tar.TypeLink:
		target, ok := f.index.byName[entry.Target]
		if !ok {
			return nil, &fs.PathError{Op: "open", Path: requestedName, Err: fmt.Errorf("%w: unresolved hardlink", ErrCorruptTarIndex)}
		}
		// Report the link entry's own metadata but read the target's bytes.
		file := f.openRegular(target)
		file.info = indexedTarEntryInfo{entry: entry}
		return file, nil

	case tar.TypeSymlink:
		target, err := f.followSymlink(entry)
		if err != nil {
			return nil, &fs.PathError{Op: "open", Path: requestedName, Err: err}
		}
		return f.openEntry(requestedName, target)

	default:
		// Character/block devices and FIFOs have no fs.FS representation.
		return nil, &fs.PathError{Op: "open", Path: requestedName, Err: ErrUnsupportedTarEntry}
	}
}

// openRegular creates a fresh SectionReader over the entry's body. A new
// reader per Open guarantees independent file offsets.
func (f *IndexedTarFS) openRegular(entry *TarEntry) *indexedTarFile {
	return &indexedTarFile{
		sr:   io.NewSectionReader(f.source, entry.BodyOff, entry.Size),
		info: indexedTarEntryInfo{entry: entry},
	}
}

func (f *IndexedTarFS) openRoot() (fs.File, error) {
	return newIndexedDirFile(implicitDirInfo{implicitDirEntry{name: "."}}, f.index.dirs["."]), nil
}

// followSymlink resolves an in-archive symlink chain. Absolute targets and
// targets that normalize outside the archive root are rejected; cycles and
// dangling links are reported as errors.
func (f *IndexedTarFS) followSymlink(start *TarEntry) (*TarEntry, error) {
	visited := make(map[string]bool)
	current := start
	for {
		if visited[current.Name] {
			return nil, fmt.Errorf("%w: %s", ErrHardlinkCycle, start.Name)
		}
		visited[current.Name] = true

		target := current.Linkname
		var resolved string
		if path.IsAbs(target) {
			// Disallow escaping the archive namespace via absolute links.
			resolved = path.Clean(target)
			if !fs.ValidPath(resolved) {
				return nil, fs.ErrInvalid
			}
		} else {
			resolved = path.Clean(path.Join(path.Dir(current.Name), target))
			if !fs.ValidPath(resolved) {
				return nil, fs.ErrInvalid
			}
		}

		next, ok := f.index.byName[resolved]
		if !ok {
			return nil, fs.ErrNotExist
		}
		if next.Typeflag != tar.TypeSymlink {
			return next, nil
		}
		current = next
	}
}

// ReadDir reads the named directory and returns its sorted entries, serving
// entirely from the prebuilt index.
func (f *IndexedTarFS) ReadDir(name string) ([]fs.DirEntry, error) {
	if err := f.checkErr(); err != nil {
		return nil, err
	}
	if !fs.ValidPath(name) {
		return nil, &fs.PathError{Op: "readdir", Path: name, Err: fs.ErrInvalid}
	}
	archiveName := path.Join(f.prefix, name)

	if entry, ok := f.index.byName[archiveName]; ok && entry.Typeflag != tar.TypeDir {
		return nil, &fs.PathError{Op: "readdir", Path: name, Err: errors.New("not a directory")}
	}
	entries, ok := f.index.dirs[archiveName]
	if !ok {
		return nil, &fs.PathError{Op: "readdir", Path: name, Err: fs.ErrNotExist}
	}
	return entries, nil
}

// Stat returns the entry's own metadata without following symlinks (lstat
// semantics). Implicit directories are reported as synthesized directory
// infos.
func (f *IndexedTarFS) Stat(name string) (fs.FileInfo, error) {
	if err := f.checkErr(); err != nil {
		return nil, err
	}
	if !fs.ValidPath(name) {
		return nil, &fs.PathError{Op: "stat", Path: name, Err: fs.ErrInvalid}
	}
	archiveName := path.Join(f.prefix, name)

	if archiveName == "." {
		return implicitDirInfo{implicitDirEntry{name: "."}}, nil
	}
	if entry, ok := f.index.byName[archiveName]; ok {
		return indexedTarEntryInfo{entry: entry}, nil
	}
	if _, ok := f.index.dirs[archiveName]; ok {
		return implicitDirInfo{implicitDirEntry{name: path.Base(archiveName)}}, nil
	}
	return nil, &fs.PathError{Op: "stat", Path: name, Err: fs.ErrNotExist}
}

// Sub returns an fs.FS rooted at the named directory. It shares the same
// source and index and preserves any construction-time error root instead
// of resetting it.
func (f *IndexedTarFS) Sub(dir string) (fs.FS, error) {
	// A sticky construction-time error must survive Sub layering: hand back
	// another view carrying the same root error instead of resetting it.
	if f.rootErr != nil {
		sub := *f
		sub.prefix = path.Join(f.prefix, dir)
		return &sub, nil
	}
	if !fs.ValidPath(dir) {
		return nil, &fs.PathError{Op: "sub", Path: dir, Err: fs.ErrInvalid}
	}
	info, err := f.Stat(dir)
	if err != nil {
		return nil, err
	}
	if !info.IsDir() {
		return nil, &fs.PathError{Op: "sub", Path: dir, Err: errors.New("not a directory")}
	}
	sub := *f
	sub.prefix = path.Join(f.prefix, dir)
	return &sub, nil
}

// ReadLink returns the raw target of the named symbolic link. Hardlinks are
// not symbolic links and are not exposed through this method.
func (f *IndexedTarFS) ReadLink(name string) (string, error) {
	if err := f.checkErr(); err != nil {
		return "", err
	}
	if !fs.ValidPath(name) {
		return "", &fs.PathError{Op: "readlink", Path: name, Err: fs.ErrInvalid}
	}
	archiveName := path.Join(f.prefix, name)
	entry, ok := f.index.byName[archiveName]
	if !ok {
		return "", &fs.PathError{Op: "readlink", Path: name, Err: fs.ErrNotExist}
	}
	if entry.Typeflag != tar.TypeSymlink {
		return "", &fs.PathError{Op: "readlink", Path: name, Err: fs.ErrInvalid}
	}
	return entry.Linkname, nil
}

// Lstat returns the named entry's own metadata without following symlinks,
// satisfying the Lstat method of fs.ReadLinkFS.
func (f *IndexedTarFS) Lstat(name string) (fs.FileInfo, error) { return f.Stat(name) }

// ReadFile reads the complete contents of the named regular file straight
// from its indexed body range.
func (f *IndexedTarFS) ReadFile(name string) ([]byte, error) {
	file, err := f.Open(name)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	return io.ReadAll(file)
}

// Close is a no-op. The source io.ReaderAt is owned by the caller and is
// never closed by the indexed file system.
func (f *IndexedTarFS) Close() error { return nil }

// checkErr returns the sticky construction-time error, if any, so every
// operation fails consistently even after Sub has layered more prefixes.
func (f *IndexedTarFS) checkErr() error {
	if f.rootErr != nil {
		return f.rootErr
	}
	if f.index == nil {
		return ErrCorruptTarIndex
	}
	return nil
}

// indexedDirFile is an open directory with its own paging cursor, so each
// Open returns an independent handle. ReadDir(-1) returns the remaining
// entries and reports an empty listing (not the whole directory again) once
// exhausted, matching os.File semantics.
type indexedDirFile struct {
	info    fs.FileInfo
	entries []fs.DirEntry
	pos     int
}

func newIndexedDirFile(info fs.FileInfo, entries []fs.DirEntry) *indexedDirFile {
	// Copy the backing slice so callers can never mutate index state through
	// a file handle.
	return &indexedDirFile{info: info, entries: slices.Clone(entries)}
}

func (d *indexedDirFile) Stat() (fs.FileInfo, error) { return d.info, nil }
func (d *indexedDirFile) Read([]byte) (int, error) {
	return 0, errors.New("cannot read a directory file")
}
func (d *indexedDirFile) Close() error { return nil }

// ReadDir implements fs.ReadDirFile with standard paging semantics.
func (d *indexedDirFile) ReadDir(n int) ([]fs.DirEntry, error) {
	if n <= 0 {
		if d.pos >= len(d.entries) {
			return nil, nil
		}
		out := d.entries[d.pos:]
		d.pos = len(d.entries)
		return out, nil
	}
	if d.pos >= len(d.entries) {
		return nil, io.EOF
	}
	end := min(d.pos+n, len(d.entries))
	out := d.entries[d.pos:end]
	d.pos = end
	return out, nil
}

// indexedTarFile is an open regular file (or hardlink view). It wraps a
// fresh SectionReader so every handle has an independent sequential offset,
// and implements io.ReaderAt with the standard EOF contract.
type indexedTarFile struct {
	sr   *io.SectionReader
	info fs.FileInfo
}

func (f *indexedTarFile) Stat() (fs.FileInfo, error) { return f.info, nil }
func (f *indexedTarFile) Read(p []byte) (int, error) { return f.sr.Read(p) }

// ReadAt reads at an absolute logical offset. Per the io.ReaderAt contract,
// reads starting at or past the end return (0, io.EOF), and reads spanning
// the end return the available bytes together with io.EOF.
func (f *indexedTarFile) ReadAt(p []byte, off int64) (int, error) { return f.sr.ReadAt(p, off) }

// Seek supports independent per-handle positioning.
func (f *indexedTarFile) Seek(offset int64, whence int) (int64, error) {
	return f.sr.Seek(offset, whence)
}

// Close releases no resources: the source is caller-owned.
func (f *indexedTarFile) Close() error { return nil }

// indexedTarEntryInfo presents a TarEntry as both fs.FileInfo and
// fs.DirEntry using only portable metadata.
type indexedTarEntryInfo struct {
	entry *TarEntry
}

func (i indexedTarEntryInfo) Name() string { return path.Base(i.entry.Name) }

// Size reports the logical size. Hardlinks have no physical body of their
// own and share the resolved target's size.
func (i indexedTarEntryInfo) Size() int64 {
	if i.entry.Typeflag == tar.TypeLink {
		return i.entry.TargetSize
	}
	return i.entry.Size
}
func (i indexedTarEntryInfo) Mode() fs.FileMode  { return fs.FileMode(i.entry.Mode) }
func (i indexedTarEntryInfo) ModTime() time.Time { return i.entry.ModTime }
func (i indexedTarEntryInfo) IsDir() bool        { return i.Mode().IsDir() }
func (i indexedTarEntryInfo) Sys() any           { return nil }
func (i indexedTarEntryInfo) Type() fs.FileMode  { return i.Mode().Type() }
func (i indexedTarEntryInfo) Info() (fs.FileInfo, error) {
	return i, nil
}

// Interface guards.
var (
	_ fs.FS         = (*IndexedTarFS)(nil)
	_ fs.ReadDirFS  = (*IndexedTarFS)(nil)
	_ fs.StatFS     = (*IndexedTarFS)(nil)
	_ fs.SubFS      = (*IndexedTarFS)(nil)
	_ fs.ReadFileFS = (*IndexedTarFS)(nil)
	_ fs.ReadLinkFS = (*IndexedTarFS)(nil)

	_ fs.ReadDirFile = (*indexedDirFile)(nil)

	_ fs.File     = (*indexedTarFile)(nil)
	_ io.ReaderAt = (*indexedTarFile)(nil)
	_ io.Seeker   = (*indexedTarFile)(nil)
)
