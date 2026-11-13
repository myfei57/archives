package archives

import (
	"archive/tar"
	"context"
	"fmt"
	"io"
	"io/fs"
	"path"
	"slices"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/klauspost/compress/zip"
)

// CanonicalModePolicy decides how the permission and special mode bits of
// archive entries are determined.
type CanonicalModePolicy int

const (
	// CanonicalModePreserve keeps each input entry's own permission bits
	// (and setuid/setgid/sticky bits). File type bits (regular, directory,
	// symlink) are always preserved regardless of the policy.
	CanonicalModePreserve CanonicalModePolicy = iota

	// CanonicalModeFixed replaces the permission and special bits of every
	// entry with Metadata.FixedMode, while file type bits are retained from
	// the input entry.
	CanonicalModeFixed
)

// CanonicalHardlinkPolicy decides what happens with hard links when the
// target archive format cannot represent them. TAR supports hard links
// natively, so this policy only affects ZIP output.
type CanonicalHardlinkPolicy int

const (
	// CanonicalHardlinkReject aborts the archive with an error before any
	// output byte is written when a hard link is encountered.
	CanonicalHardlinkReject CanonicalHardlinkPolicy = iota

	// CanonicalHardlinkMaterialize writes every hard link as an independent
	// regular file, each carrying its own copy of the shared content. The
	// resulting archive no longer records the link relationship.
	CanonicalHardlinkMaterialize
)

// CanonicalMetadata is the explicit, deterministic metadata policy applied
// uniformly to every entry of a [CanonicalArchive]. Zero values are
// deliberate and are written as zero; unlike the loose defaults of the
// [Tar] type (where, for example, a Uid of 0 historically means "leave the
// host value untouched"), every field here is an explicit instruction.
type CanonicalMetadata struct {
	// ModTime is written to every entry. A zero time is a valid, explicit
	// choice and yields entries carrying no meaningful timestamp.
	ModTime time.Time

	// ModePolicy selects how permission bits are determined.
	ModePolicy CanonicalModePolicy

	// FixedMode is used when ModePolicy is [CanonicalModeFixed]. Only the
	// permission and setuid/setgid/sticky bits are taken from it; file type
	// bits always come from the corresponding input entry.
	FixedMode fs.FileMode

	// Uid and Gid are written to every entry unconditionally, including
	// zero values. They never fall back to host-derived ids.
	Uid int
	Gid int

	// Uname and Gname are written to TAR entries when NumericOwner is false.
	// They are not represented in the ZIP format and are ignored there.
	Uname string
	Gname string

	// NumericOwner omits owner names and keeps only the numeric Uid/Gid.
	NumericOwner bool
}

// CanonicalArchive produces byte-for-byte reproducible (deterministic)
// archives in either TAR or ZIP form.
//
// Given the same logical set of files, the same version of Go and of the
// archive codec libraries, and an identical CanonicalArchive configuration,
// Archive always emits the same bytes. In particular, the result does not
// depend on:
//
//   - the order of the input slice or the iteration order of any input map,
//   - host file modification times, host uid/gid values, or host-specific
//     FileInfo.Sys data,
//   - the traversal order in which [FilesFromDisk] happened to encounter
//     files sharing an inode.
//
// Before any output byte is written, all entry paths are normalized and
// validated (slash-separated, relative, no ".." escapes, no NUL bytes),
// duplicate paths and post-cleaning path collisions are rejected, and
// unsupported entry types are rejected. Entries are then emitted sorted by
// canonical path.
//
// Hard links are re-derived from the input FileInfo rather than trusted from
// [FilesFromDisk]: within each set of regular files sharing an inode, the
// lexicographically smallest canonical path becomes the single content
// anchor, and the other members reference it.
//
// The determinism guarantee covers the TAR/ZIP container bytes only. It does
// NOT extend to outer compression streams produced by arbitrary third-party
// compressors (gzip, zstd, xz, and similar wrappers), which may add their own
// headers, timestamps, or version-dependent framing on top of the container.
//
// The input []FileInfo and its FileInfo values are never modified in place.
// On failure, Archive returns a non-nil error and the output must be
// discarded; it never returns nil with a partial or corrupt archive.
type CanonicalArchive struct {
	// Template defines the output container and its format-specific settings.
	// It must be a [Tar] or [Zip] value (or a non-nil pointer to one).
	//
	// For Tar, the Format (and deprecated FormatGNU) field selects the TAR
	// format; when unset, PAX is used so long names, large files and UTF-8
	// are always representable. For Zip, SelectiveCompression and
	// Compression determine the per-entry method. All other template fields
	// (including Uid/Gid/Uname/Gname and ContinueOnError) are ignored:
	// canonical mode is always strict and driven by [CanonicalMetadata].
	Template any

	// Metadata is the explicit metadata policy applied to every entry.
	Metadata CanonicalMetadata

	// Hardlinks controls hard-link handling for formats that cannot
	// represent them natively (ZIP). It is ignored for TAR output.
	Hardlinks CanonicalHardlinkPolicy

	// hardlinkKeyFunc overrides inode detection. It is unexported and
	// intended for deterministic in-process testing on platforms where
	// real inode information is unavailable.
	hardlinkKeyFunc func(fs.FileInfo) (canonicalInodeKey, bool)
}

// Archive writes a canonical, deterministic archive of files to output.
// It implements the [Archiver] interface.
func (ca CanonicalArchive) Archive(ctx context.Context, output io.Writer, files []FileInfo) error {
	if output == nil {
		return fmt.Errorf("canonical archive: output writer is nil")
	}

	entries, err := ca.prepare(ctx, files)
	if err != nil {
		// Every failure up to this point happens before the first output byte.
		return fmt.Errorf("canonical archive: %w", err)
	}

	switch tpl := ca.Template.(type) {
	case Tar:
		return ca.archiveTar(ctx, output, tpl, entries)
	case *Tar:
		if tpl == nil {
			return fmt.Errorf("canonical archive: nil Tar template")
		}
		return ca.archiveTar(ctx, output, *tpl, entries)
	case Zip:
		return ca.archiveZip(ctx, output, tpl, entries)
	case *Zip:
		if tpl == nil {
			return fmt.Errorf("canonical archive: nil Zip template")
		}
		return ca.archiveZip(ctx, output, *tpl, entries)
	case nil:
		return fmt.Errorf("canonical archive: Template must be set to Tar or Zip")
	default:
		return fmt.Errorf("canonical archive: unsupported template type %T; want Tar or Zip", ca.Template)
	}
}

// canonicalEntry is one fully validated, normalized input to the format
// writers. It never aliases the caller's FileInfo in a way that could mutate
// the original: file is an independent shallow copy with a wrapped FileInfo.
type canonicalEntry struct {
	file     FileInfo
	name     string // cleaned, slash-separated, no trailing slash
	dir      bool
	symlink  bool
	hardlink bool   // regular file that is a non-anchor member of an inode set
	anchor   string // canonical name of the hard-link content anchor
}

// canonicalFileInfo presents deterministic mode and modification time to the
// standard archive header builders while hiding host-specific Sys data, so
// nothing from the host (uid/gid, atime/ctime, device numbers, host header
// blobs) can leak into freshly built headers.
type canonicalFileInfo struct {
	fs.FileInfo
	mode    fs.FileMode
	modTime time.Time
}

func (c canonicalFileInfo) Mode() fs.FileMode  { return c.mode }
func (c canonicalFileInfo) ModTime() time.Time { return c.modTime }
func (canonicalFileInfo) Sys() any             { return nil }

// prepare runs the complete validation and normalization pipeline before any
// output is produced.
func (ca CanonicalArchive) prepare(ctx context.Context, files []FileInfo) ([]canonicalEntry, error) {
	entries := make([]canonicalEntry, 0, len(files))
	indexByName := make(map[string]int)
	inodeSets := make(map[canonicalInodeKey][]int)

	keyFor := ca.hardlinkKeyFunc
	if keyFor == nil {
		keyFor = defaultHardlinkKey
	}

	for i := range files {
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		in := files[i] // never mutated; all derived data goes into copies

		name, err := canonicalArchivePath(in.NameInArchive)
		if err != nil {
			return nil, fmt.Errorf("file %d: %w", i, err)
		}

		mode := in.Mode()
		switch {
		case mode.IsRegular():
		case mode.IsDir():
		case mode&fs.ModeSymlink != 0:
		default:
			return nil, fmt.Errorf("file %d (%s): canonical archive supports only regular files, directories, and symbolic links; mode %v is not representable in both tar and zip", i, name, mode)
		}

		if prior, dup := indexByName[name]; dup {
			return nil, fmt.Errorf("duplicate archive path %q after normalization (also used by file %d)", name, prior)
		}

		if in.Size() < 0 {
			return nil, fmt.Errorf("file %d (%s): negative declared size %d", i, name, in.Size())
		}

		// Determine the canonical mode: file type bits always survive; the
		// permission/special bits follow the explicit policy.
		var canonMode fs.FileMode
		switch ca.Metadata.ModePolicy {
		case CanonicalModeFixed:
			canonMode = mode&fs.ModeType | ca.Metadata.FixedMode&(fs.ModePerm|fs.ModeSetuid|fs.ModeSetgid|fs.ModeSticky)
		default: // CanonicalModePreserve
			canonMode = mode & (fs.ModeType | fs.ModePerm | fs.ModeSetuid | fs.ModeSetgid | fs.ModeSticky)
		}

		out := in // shallow copy of the value; original stays untouched
		out.FileInfo = canonicalFileInfo{
			FileInfo: in.FileInfo,
			mode:     canonMode,
			modTime:  ca.Metadata.ModTime,
		}
		out.Header = nil // fresh headers only; never reuse host/format header blobs

		e := canonicalEntry{file: out, name: name}
		switch {
		case mode.IsDir():
			e.dir = true
		case mode&fs.ModeSymlink != 0:
			e.symlink = true
			// Symlink targets are opaque strings and are kept verbatim on the copy.
		case mode.IsRegular():
			// Inode detection must see the ORIGINAL Sys data; our wrapper hides it.
			if key, ok := keyFor(in.FileInfo); ok {
				inodeSets[key] = append(inodeSets[key], len(entries))
			}
		}

		indexByName[name] = len(entries)
		entries = append(entries, e)
	}

	// Structural collision: a non-directory entry must never be required to
	// act as a directory of another entry.
	for _, e := range entries {
		if e.dir {
			continue
		}
		parent := path.Dir(e.name)
		for parent != "." && parent != "/" {
			if idx, ok := indexByName[parent]; ok && !entries[idx].dir {
				return nil, fmt.Errorf("path collision: %q is a non-directory entry but is also required to be a directory for %q", parent, e.name)
			}
			parent = path.Dir(parent)
		}
	}

	// Hard-link sets: the lexicographically smallest canonical path is the
	// content anchor. This is independent of input or traversal order.
	for _, members := range inodeSets {
		if len(members) < 2 {
			continue
		}
		anchorIdx := members[0]
		for _, idx := range members[1:] {
			if entries[idx].name < entries[anchorIdx].name {
				anchorIdx = idx
			}
		}
		for _, idx := range members {
			if idx == anchorIdx {
				continue // the anchor keeps its body
			}
			entries[idx].hardlink = true
			entries[idx].anchor = entries[anchorIdx].name
			entries[idx].file.LinkTarget = entries[anchorIdx].name
		}
	}

	// Total, stable ordering independent of input enumeration order.
	slices.SortStableFunc(entries, func(a, b canonicalEntry) int {
		return strings.Compare(a.name, b.name)
	})

	// Every regular entry (including materialized hard links) must be able
	// to supply content; validate before writing anything.
	for _, e := range entries {
		if !e.dir && !e.symlink && e.file.Open == nil {
			return nil, fmt.Errorf("file %q requires content but has no Open callback", e.name)
		}
	}

	return entries, nil
}

// canonicalArchivePath normalizes and validates a single in-archive path.
func canonicalArchivePath(raw string) (string, error) {
	if raw == "" {
		return "", fmt.Errorf("empty archive path")
	}
	if strings.ContainsRune(raw, 0) {
		return "", fmt.Errorf("archive path %q contains a NUL byte", raw)
	}
	// Archive paths are slash-separated on every host OS.
	name := strings.ReplaceAll(raw, "\\", "/")
	cleaned := path.Clean(name)
	if cleaned == "." || cleaned == "" {
		return "", fmt.Errorf("archive path %q resolves to an invalid name %q", raw, cleaned)
	}
	if path.IsAbs(cleaned) {
		return "", fmt.Errorf("archive path %q must not be absolute", raw)
	}
	// After path.Clean, any surviving ".." can only be a leading parent
	// reference, which would escape the archive root on extraction.
	if cleaned == ".." || strings.HasPrefix(cleaned, "../") {
		return "", fmt.Errorf("archive path %q must not contain parent-directory (..) components", raw)
	}
	return cleaned, nil
}

// archiveTar writes the prepared entries as a deterministic TAR stream.
func (ca CanonicalArchive) archiveTar(ctx context.Context, output io.Writer, tpl Tar, entries []canonicalEntry) (retErr error) {
	tw := tar.NewWriter(output)
	defer func() {
		if closeErr := tw.Close(); retErr == nil && closeErr != nil {
			retErr = fmt.Errorf("canonical tar: finalizing archive: %w", closeErr)
		}
	}()

	format := tpl.Format
	if format == 0 {
		if tpl.FormatGNU {
			format = tar.FormatGNU
		} else {
			// PAX uniformly supports long names, large files and UTF-8 without
			// dropping semantics; USTAR-style blocks are still emitted whenever
			// the entry fits. PAX record keys are emitted sorted by the
			// standard library, so extension headers stay deterministic.
			format = tar.FormatPAX
		}
	}

	for _, e := range entries {
		if err := ctx.Err(); err != nil {
			return err
		}

		// A nil Sys on the wrapped FileInfo guarantees no host-derived uid/gid,
		// atime/ctime or device numbers reach the header.
		hdr, err := tar.FileInfoHeader(e.file.FileInfo, e.file.LinkTarget)
		if err != nil {
			return fmt.Errorf("canonical tar: %s: creating header: %w", e.name, err)
		}
		hdr.Name = e.name // canonical path; FileInfoHeader only knew the base name

		if e.hardlink {
			hdr.Typeflag = tar.TypeLink
			hdr.Linkname = e.anchor
			hdr.Size = 0
		}

		// Explicit, uniform format and metadata for every entry.
		hdr.Format = format
		hdr.ModTime = ca.Metadata.ModTime
		hdr.AccessTime = time.Time{}
		hdr.ChangeTime = time.Time{}
		// Zero is an explicit value: no "Uid != 0" guard.
		hdr.Uid = ca.Metadata.Uid
		hdr.Gid = ca.Metadata.Gid
		if ca.Metadata.NumericOwner {
			hdr.Uname = ""
			hdr.Gname = ""
		} else {
			hdr.Uname = ca.Metadata.Uname
			hdr.Gname = ca.Metadata.Gname
		}
		// Never carry host-derived extension records or xattrs.
		hdr.PAXRecords = nil
		hdr.Xattrs = nil
		hdr.Devmajor, hdr.Devminor = 0, 0

		if err := tw.WriteHeader(hdr); err != nil {
			return fmt.Errorf("canonical tar: %s: writing header: %w", e.name, err)
		}

		// Directories, symlinks and hard links carry no body; iteration
		// continues with the remaining entries.
		if hdr.Typeflag != tar.TypeReg {
			continue
		}
		if err := copyCanonicalBody(e.file, tw, e.name); err != nil {
			return err
		}
	}
	return nil
}

// archiveZip writes the prepared entries as a deterministic ZIP stream.
func (ca CanonicalArchive) archiveZip(ctx context.Context, output io.Writer, tpl Zip, entries []canonicalEntry) (retErr error) {
	// ZIP-specific validation, still before any output byte.
	for _, e := range entries {
		if !utf8.ValidString(e.name) {
			return fmt.Errorf("canonical zip: %s: archive path is not valid UTF-8", e.name)
		}
		if len(e.name) > 65535 {
			return fmt.Errorf("canonical zip: %s: archive path exceeds the 65535 byte limit", e.name)
		}
		if e.hardlink && ca.Hardlinks == CanonicalHardlinkReject {
			return fmt.Errorf("canonical zip: %s: hard links are not representable in zip; set Hardlinks to CanonicalHardlinkMaterialize to store independent content copies, or use tar", e.name)
		}
	}

	zw := zip.NewWriter(output)
	defer func() {
		if closeErr := zw.Close(); retErr == nil && closeErr != nil {
			retErr = fmt.Errorf("canonical zip: finalizing archive: %w", closeErr)
		}
	}()

	for _, e := range entries {
		if err := ctx.Err(); err != nil {
			return err
		}

		hdr, err := zip.FileInfoHeader(e.file.FileInfo)
		if err != nil {
			return fmt.Errorf("canonical zip: %s: creating header: %w", e.name, err)
		}
		hdr.Name = e.name
		if e.dir {
			hdr.Name += "/" // required by the zip writer for directory entries
		}

		// Normalize every variable, potentially host-derived header field.
		hdr.Comment = ""
		hdr.Extra = nil // the writer re-adds only deterministic extras (extended time, zip64)
		hdr.NonUTF8 = false
		hdr.Flags &^= 0x800
		if !isASCII([]byte(hdr.Name)) {
			hdr.Flags |= 0x800 // explicit UTF-8 indicator for non-ASCII names
		}
		hdr.SetModTime(ca.Metadata.ModTime.UTC())
		if ca.Metadata.ModTime.IsZero() {
			// SetModTime encodes the zero time as invalid (wrapped) MS-DOS
			// values; an explicit zero-time policy must leave no timestamp.
			hdr.Modified = time.Time{}
			hdr.ModifiedDate = 0
			hdr.ModifiedTime = 0
		}

		// Deterministic method selection.
		if e.dir {
			hdr.Method = zip.Store
		} else if tpl.SelectiveCompression {
			if _, ok := compressedFormats[strings.ToLower(path.Ext(e.name))]; ok {
				hdr.Method = zip.Store
			} else {
				hdr.Method = tpl.Compression
			}
		} else {
			hdr.Method = tpl.Compression
		}

		// CreateHeader takes ownership of hdr and may mutate its own copy;
		// the caller's FileInfo is already isolated from it.
		w, err := zw.CreateHeader(hdr)
		if err != nil {
			return fmt.Errorf("canonical zip: %s: writing header: %w", e.name, err)
		}

		switch {
		case e.symlink:
			// Symlinks are stored as the link target string, matching Zip.Archive.
			if _, err := io.WriteString(w, e.file.LinkTarget); err != nil {
				return fmt.Errorf("canonical zip: %s: writing link target: %w", e.name, err)
			}
			continue
		case e.dir:
			continue // directory entries have no body
		default:
			// Regular file, including a materialized hard link: each gets its
			// own independent content copy.
			if err := copyCanonicalBody(e.file, w, e.name); err != nil {
				return err
			}
		}
	}
	return nil
}

// copyCanonicalBody streams a regular file's content to w, opening the file
// lazily and failing if the actual content does not match its declared size.
func copyCanonicalBody(file FileInfo, w io.Writer, name string) error {
	if file.Open == nil {
		return fmt.Errorf("canonical archive: %s: cannot read content: no Open callback", name)
	}
	r, err := file.Open()
	if err != nil {
		return fmt.Errorf("canonical archive: %s: opening content: %w", name, err)
	}
	defer r.Close()

	declared := file.Size()
	if declared < 0 {
		return fmt.Errorf("canonical archive: %s: negative declared size %d", name, declared)
	}

	// Cap writes at the declared size so an over-long stream can never
	// corrupt a tar block or silently resize a zip entry.
	n, err := io.Copy(w, io.LimitReader(r, declared))
	if err != nil {
		return fmt.Errorf("canonical archive: %s: copying content: %w", name, err)
	}
	if n < declared {
		return fmt.Errorf("canonical archive: %s: content ended after %d bytes but %d were declared", name, n, declared)
	}

	// Detect content beyond the declared size without writing it anywhere.
	var extra [1]byte
	if m, _ := io.ReadFull(r, extra[:]); m > 0 {
		return fmt.Errorf("canonical archive: %s: content exceeds declared size %d", name, declared)
	}
	return nil
}

// Interface guard.
var _ Archiver = CanonicalArchive{}
