package archives

import (
	"archive/tar"
	"context"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// This file implements TranscodeArchive, which converts a file set between
// the archive formats that can both read and write entries in a streaming
// fashion: Tar and Zip. The conversion never materializes the source archive
// into a regular directory.
//
// The central correctness constraint is the lifetime of source entries: an
// Extractor's FileInfo.Open is only guaranteed to be valid while the extract
// handler is running (for example, Tar's Open hands out the current
// tar.Reader, which advances to the next entry as soon as the handler
// returns). TranscodeArchive therefore either:
//
//   - consumes an entry's content inline, before the handler returns, by
//     handing it to the target through an ArchiveAsync job and waiting for
//     the result, or
//   - copies the content into a private, byte-budgeted spool that it owns,
//     after which the entry can be re-opened independently of the source
//     stream (this is required to materialize hard links, including forward
//     references).
//
// All temporary spool resources are removed before Transcode returns, and the
// caller's input/output streams are never closed by this package.

// LinkHandling decides what happens to a link entry that the target format
// cannot represent in its original form.
type LinkHandling int

const (
	// LinkPreserve keeps the link semantics (a link stays a link). When the
	// target cannot encode that kind of link, the entry is rejected with an
	// explicit error rather than being silently converted.
	LinkPreserve LinkHandling = iota

	// LinkMaterialize replaces the link with a regular file whose content is
	// copied from the linked entry. Resolution happens strictly inside the
	// archive: entries on disk are never opened or followed. A forward
	// reference that cannot be resolved within the spool budget fails the
	// whole operation explicitly; an empty placeholder file is never written.
	LinkMaterialize

	// LinkReject fails with an explicit error when such a link is encountered.
	LinkReject
)

// ConflictHandling decides what happens when two source entries map to the
// same normalized name in the target archive.
type ConflictHandling int

const (
	// ConflictError (the default) aborts with an error on the first duplicate.
	ConflictError ConflictHandling = iota

	// ConflictSkip keeps the first entry and skips later duplicates (they are
	// counted as skipped in the result).
	ConflictSkip

	// ConflictSuffix keeps both entries by renaming later duplicates with a
	// numeric suffix, e.g. "a.txt", "a-1.txt", "a-2.txt".
	ConflictSuffix
)

// MetadataHandling decides what happens to entry metadata the target format
// cannot encode. Portable metadata (entry type, permission bits, modification
// time and size) is always retained.
type MetadataHandling int

const (
	// MetadataPreserve keeps every portable attribute and skips entries whose
	// very type the target cannot represent, marking the result incomplete.
	// Attributes the target simply has no field for (for example Unix UID/GID
	// in a Zip) are dropped by the target header conversion.
	MetadataPreserve MetadataHandling = iota

	// MetadataReject fails with an explicit error when non-portable metadata
	// or an unsupported entry type is encountered.
	MetadataReject
)

// Sentinel errors returned (possibly wrapped) by Transcode. Callers can use
// errors.Is to detect the specific limit or policy that was hit.
var (
	ErrTranscodeUnsupportedCombo = errors.New("transcode: unsupported source/target combination; only Tar and Zip are supported")
	ErrMaxEntries                = errors.New("transcode: maximum number of entries exceeded")
	ErrSpoolLimit                = errors.New("transcode: private spool budget exhausted")
	ErrDecompressedLimit         = errors.New("transcode: maximum decompressed byte count exceeded")
	ErrNameConflict              = errors.New("transcode: duplicate entry name after normalization")
	ErrUnsupportedLink           = errors.New("transcode: link cannot be represented by the target format under the configured policy")
	ErrUnsupportedMetadata       = errors.New("transcode: entry metadata or type cannot be represented by the target format")
	ErrDanglingLink              = errors.New("transcode: link target was not found among the source entries")
	ErrInvalidEntryName          = errors.New("transcode: invalid entry name")

	// errTargetStopped is used internally to wake a blocked source handler
	// after the target writer goroutine has exited.
	errTargetStopped = errors.New("transcode: target writer has stopped")
)

// TranscodeArchive converts the entries of a source archive into a target
// archive without extracting them to a directory first.
//
// Source must be a Tar or Zip Extractor and Target must be the Tar or Zip
// ArchiverAsync; every Tar/Zip pairing is supported. A Zip source additionally
// requires the input stream to implement io.ReaderAt and io.Seeker, as Zip
// extraction always has.
//
// The zero value is not usable: Source and Target must be set. Limits set to
// zero mean "unlimited", except MaxConcurrency, where zero means one.
type TranscodeArchive struct {
	// Source reads entries from the input archive.
	Source Extractor

	// Target writes entries to the output archive. It must implement
	// ArchiverAsync (Tar and Zip do).
	Target ArchiverAsync

	// MaxEntries is the hard upper bound on the number of source entries
	// walked, including entries that are later filtered out. Zero means
	// unlimited.
	MaxEntries int64

	// MaxSpoolBytes is the hard upper bound on the total number of
	// decompressed bytes held in the private spool at any time. The spool is
	// only used when link materialization requires retaining content beyond
	// the source handler's lifetime. Zero means unlimited.
	MaxSpoolBytes int64

	// MaxDecompressedBytes is the hard upper bound on the total number of
	// decompressed bytes read from source entries. Zero means unlimited.
	MaxDecompressedBytes int64

	// MaxConcurrency bounds how many target jobs may be in flight at once.
	// Entries whose content is borrowed from the sequential source stream are
	// always completed before the source handler returns, so a value greater
	// than one only pipelines spool-backed entries. Zero means one.
	MaxConcurrency int

	// Filter, if set, is called for every source entry in walk order.
	// Return ok=false to skip the entry. Otherwise newName becomes the
	// entry's name in the target archive; it is normalized afterwards and
	// checked for conflicts like any other name.
	Filter func(info FileInfo) (newName string, ok bool)

	// OnConflict selects the duplicate-name policy. Defaults to ConflictError.
	OnConflict ConflictHandling

	// Hardlinks selects the policy for hard links (only Tar can encode them).
	// Defaults to LinkPreserve.
	Hardlinks LinkHandling

	// Symlinks selects the policy for symbolic links. Both Tar and Zip can
	// preserve symbolic link semantics, which is the default. Materialization
	// never follows paths on disk; only in-archive entries are considered.
	Symlinks LinkHandling

	// Metadata selects the policy for metadata or entry types the target
	// cannot encode. Defaults to MetadataPreserve.
	Metadata MetadataHandling

	// ContinueOnError, when true, records per-entry errors in the result and
	// keeps processing remaining entries. The returned result is marked
	// Incomplete. Errors that finalize the target archive or violate a hard
	// limit are always returned regardless of this setting. Context
	// cancellation always aborts.
	ContinueOnError bool
}

// TranscodeResult reports what a Transcode run did.
type TranscodeResult struct {
	// EntriesSeen is the number of entries walked from the source.
	EntriesSeen int64

	// EntriesWritten is the number of entries handed to the target writer.
	EntriesWritten int64

	// EntriesSkipped is the number of entries excluded by the filter, the
	// conflict policy, or a best-effort metadata policy.
	EntriesSkipped int64

	// BytesCopied is the total number of decompressed bytes read from source
	// entry bodies.
	BytesCopied int64

	// SpoolBytesPeak is the largest number of bytes held in the private
	// spool at any point during the run.
	SpoolBytesPeak int64

	// Incomplete is true when the output archive does not contain every
	// source entry, either because of ContinueOnError or because entries had
	// to be skipped under a best-effort policy.
	Incomplete bool

	// Errors holds the per-entry errors that were tolerated while
	// ContinueOnError was enabled.
	Errors []error
}

// Transcode reads every entry from input and writes the converted archive to
// output. Neither stream is closed by this method. It returns when the source
// walk is finished and the target writer has fully finalized (including its
// Close/finalize error, which is always surfaced).
func (ta TranscodeArchive) Transcode(ctx context.Context, input io.Reader, output io.Writer) (*TranscodeResult, error) {
	if err := ta.validate(input); err != nil {
		return nil, err
	}
	return ta.run(ctx, input, output)
}

// validate enforces the explicitly supported Tar/Zip matrix up front.
func (ta TranscodeArchive) validate(input io.Reader) error {
	if ta.Source == nil || ta.Target == nil {
		return errors.New("transcode: both Source and Target must be configured")
	}
	if !isTarOrZip(ta.Source) {
		return fmt.Errorf("%w: unsupported source %T", ErrTranscodeUnsupportedCombo, ta.Source)
	}
	if !isTarOrZip(ta.Target) {
		return fmt.Errorf("%w: unsupported target %T", ErrTranscodeUnsupportedCombo, ta.Target)
	}
	// Zip extraction needs random access; fail early with a clear message.
	if _, ok := ta.Source.(Zip); ok || isZipPtr(ta.Source) {
		if _, ok := input.(seekReaderAt); !ok {
			return errors.New("transcode: Zip source input must implement io.ReaderAt and io.Seeker")
		}
	}
	return nil
}

func isTarOrZip(v any) bool {
	switch v.(type) {
	case Tar, *Tar, Zip, *Zip:
		return true
	}
	return false
}

func isZipPtr(v any) bool {
	_, ok := v.(*Zip)
	return ok
}

func (ta TranscodeArchive) targetIsZip() bool {
	switch ta.Target.(type) {
	case Zip, *Zip:
		return true
	}
	return false
}

// run contains the actual conversion logic. Package tests exercise it directly
// with controllable double Extractors while Transcode keeps the public
// Tar/Zip-only guarantee.
func (ta TranscodeArchive) run(ctx context.Context, input io.Reader, output io.Writer) (*TranscodeResult, error) {
	childCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	result := &TranscodeResult{}

	concurrency := ta.MaxConcurrency
	if concurrency <= 0 {
		concurrency = 1
	}
	sem := make(chan struct{}, concurrency)

	spool := &spoolManager{maxBytes: ta.MaxSpoolBytes, decompressedLimit: ta.MaxDecompressedBytes}
	defer spool.cleanup()

	// Job/result handshake. Every result channel is buffered so the target
	// goroutine can never block delivering a result after the source gave up.
	// abort is closed when the target exits with an error or the run is torn
	// down, guaranteeing that a source handler waiting on a result wakes up
	// instead of deadlocking against an exited target.
	jobs := make(chan ArchiveAsyncJob)
	abort := make(chan struct{})
	var abortOnce sync.Once
	triggerAbort := func() { abortOnce.Do(func() { close(abort) }) }

	// Capture every underlying write error, including ones a target
	// implementation might fail to surface from its finalize/Close, so that a
	// broken output stream can never be reported as success.
	captured := &errorCapturingWriter{inner: output}

	targetDone := make(chan error, 1)
	go func() {
		err := ta.Target.ArchiveAsync(childCtx, captured, jobs)
		targetDone <- err
		if err != nil {
			triggerAbort()
		}
	}()

	var errMu sync.Mutex
	recordError := func(err error) error {
		if err == nil {
			return nil
		}
		errMu.Lock()
		result.Errors = append(result.Errors, err)
		result.Incomplete = true
		errMu.Unlock()
		if ta.ContinueOnError && childCtx.Err() == nil {
			return nil
		}
		// Hard failure: stop both sides promptly.
		triggerAbort()
		cancel()
		return err
	}

	limitedOpen := func(file FileInfo) func() (fs.File, error) {
		return func() (fs.File, error) {
			opened, err := file.Open()
			if err != nil {
				return nil, err
			}
			return &limitedFSFile{
				File:   opened,
				reader: &limitedCountReader{r: opened, copied: &spool.bytesRead, limit: ta.MaxDecompressedBytes},
			}, nil
		}
	}

	seenNames := make(map[string]struct{})
	// renameMap translates normalized source names to the names actually
	// written, so hard-link targets stay valid after filtering/renaming.
	renameMap := make(map[string]string)
	// retained keyed by normalized source name: entries kept for link
	// resolution (spooled regular files and materialized aliases).
	retained := make(map[string]*spoolEntry)
	// pending holds links whose regular target has not been seen yet.
	pending := make(map[string][]pendingLink)
	var stateMu sync.Mutex

	var inflight sync.WaitGroup

	// waitResult blocks until the target acknowledges a job, the context is
	// canceled, or the target has exited. It never blocks forever.
	waitResult := func(resCh <-chan error) error {
		select {
		case err := <-resCh:
			return err
		case <-childCtx.Done():
			return childCtx.Err()
		case <-abort:
			select {
			case err := <-resCh:
				return err
			default:
				return errTargetStopped
			}
		}
	}

	acquireSem := func() error {
		select {
		case sem <- struct{}{}:
			return nil
		case <-childCtx.Done():
			return childCtx.Err()
		case <-abort:
			return errTargetStopped
		}
	}

	// emit hands one entry to the target. If borrowed is true, the entry's
	// Open is only valid for the lifetime of the current source handler, so
	// the target result is awaited inline. Spool-backed entries may pipeline
	// up to MaxConcurrency unacknowledged jobs; their arrival order at the
	// target is still the source walk order because all sends are serialized
	// through the single-threaded extract handler.
	emit := func(fi FileInfo, borrowed bool) error {
		if err := acquireSem(); err != nil {
			return err
		}
		resCh := make(chan error, 1)
		job := ArchiveAsyncJob{File: fi, Result: resCh}
		select {
		case jobs <- job:
		case <-childCtx.Done():
			<-sem
			return childCtx.Err()
		case <-abort:
			<-sem
			return errTargetStopped
		}

		if borrowed {
			defer func() { <-sem }()
			if err := waitResult(resCh); err != nil {
				return recordError(fmt.Errorf("writing entry %q: %w", fi.NameInArchive, err))
			}
			result.EntriesWritten++
			return nil
		}

		inflight.Add(1)
		go func() {
			defer inflight.Done()
			defer func() { <-sem }()
			if err := waitResult(resCh); err != nil {
				recordError(fmt.Errorf("writing entry %q: %w", fi.NameInArchive, err))
				return
			}
			atomic.AddInt64(&result.EntriesWritten, 1)
		}()
		return nil
	}

	// flushPending resolves links that were waiting for the entry named
	// targetSourceName.
	flushPending := func(targetSourceName, writtenName string, se *spoolEntry) error {
		stateMu.Lock()
		waited := pending[targetSourceName]
		delete(pending, targetSourceName)
		for _, p := range waited {
			// Materialized names alias the same retained content, so link
			// chains keep resolving.
			retained[p.outName] = se
		}
		stateMu.Unlock()

		for _, p := range waited {
			if p.preserve {
				out := p.src
				out.NameInArchive = p.outName
				out.LinkTarget = writtenName
				if err := emit(out, true); err != nil {
					return err
				}
			} else if err := emit(se.fileInfo(p.outName), false); err != nil {
				return err
			}
		}
		return nil
	}

	handler := func(hctx context.Context, file FileInfo) error {
		if err := hctx.Err(); err != nil {
			return err
		}

		seen := atomic.AddInt64(&result.EntriesSeen, 1)
		if ta.MaxEntries > 0 && seen > ta.MaxEntries {
			return recordError(fmt.Errorf("%w: saw %d entries", ErrMaxEntries, ta.MaxEntries))
		}

		// Explicit filtering and renaming.
		outName := file.NameInArchive
		if ta.Filter != nil {
			newName, ok := ta.Filter(file)
			if !ok {
				atomic.AddInt64(&result.EntriesSkipped, 1)
				return nil
			}
			outName = newName
		}

		norm, err := normalizeEntryName(outName)
		if err != nil {
			return recordError(err)
		}

		if _, exists := seenNames[norm]; exists {
			switch ta.OnConflict {
			case ConflictSkip:
				atomic.AddInt64(&result.EntriesSkipped, 1)
				return nil
			case ConflictSuffix:
				norm = uniqueSuffixName(seenNames, norm)
			default:
				return recordError(fmt.Errorf("%w: %q", ErrNameConflict, norm))
			}
		}
		seenNames[norm] = struct{}{}

		sourceName, err := normalizeEntryName(file.NameInArchive)
		if err == nil {
			renameMap[sourceName] = norm
		}

		mode := file.Mode()
		isSymlink := mode&fs.ModeSymlink != 0
		isHardlink := file.LinkTarget != "" && mode.IsRegular()
		isDir := file.IsDir()
		isSpecial := mode&(fs.ModeDevice|fs.ModeCharDevice|fs.ModeNamedPipe|fs.ModeSocket) != 0

		// Symbolic links: preserve semantics, never touch the disk.
		if isSymlink {
			switch ta.Symlinks {
			case LinkReject:
				return recordError(fmt.Errorf("%w: symbolic link %q", ErrUnsupportedLink, norm))
			case LinkMaterialize:
				targetName := resolveInArchiveName(norm, file.LinkTarget)
				stateMu.Lock()
				se, ok := retained[targetName]
				stateMu.Unlock()
				if !ok {
					return recordError(fmt.Errorf("%w: symbolic link %q -> %q cannot be materialized: target content is not retained in the spool and disk paths are never followed", ErrUnsupportedLink, norm, file.LinkTarget))
				}
				return emit(se.fileInfo(norm), false)
			default: // LinkPreserve; both Tar and Zip encode symlinks
				out := file
				out.NameInArchive = norm
				return emit(out, true)
			}
		}

		// Entry types only Tar can represent (devices, fifos, sockets).
		if isSpecial && ta.targetIsZip() {
			if ta.Metadata == MetadataReject {
				return recordError(fmt.Errorf("%w: special file %q with mode %v has no Zip representation", ErrUnsupportedMetadata, norm, mode))
			}
			errMu.Lock()
			result.Errors = append(result.Errors, fmt.Errorf("%w: special file %q with mode %v skipped for Zip target", ErrUnsupportedMetadata, norm, mode))
			result.Incomplete = true
			errMu.Unlock()
			atomic.AddInt64(&result.EntriesSkipped, 1)
			return nil
		}

		// Non-portable Unix ownership metadata for Zip targets.
		if ta.targetIsZip() && ta.Metadata == MetadataReject {
			if err := rejectUnixOwnership(file, norm); err != nil {
				return recordError(err)
			}
		}

		// Hard links: Tar encodes them natively, Zip does not.
		if isHardlink {
			targetSourceName, nerr := normalizeEntryName(file.LinkTarget)
			if nerr != nil {
				return recordError(fmt.Errorf("%w: hard link %q: %v", ErrUnsupportedLink, norm, nerr))
			}

			switch ta.Hardlinks {
			case LinkReject:
				return recordError(fmt.Errorf("%w: hard link %q -> %q", ErrUnsupportedLink, norm, file.LinkTarget))
			case LinkMaterialize:
				stateMu.Lock()
				se, ok := retained[targetSourceName]
				stateMu.Unlock()
				if ok {
					stateMu.Lock()
					retained[norm] = se
					stateMu.Unlock()
					return emit(se.fileInfo(norm), false)
				}
				// Forward reference: remember the link and fail later if the
				// target never shows up or cannot fit the spool budget.
				stateMu.Lock()
				pending[targetSourceName] = append(pending[targetSourceName], pendingLink{outName: norm, src: file})
				stateMu.Unlock()
				return nil
			default: // LinkPreserve
				if ta.targetIsZip() {
					return recordError(fmt.Errorf("%w: hard link %q -> %q: Zip has no hard-link encoding; use LinkMaterialize or LinkReject", ErrUnsupportedLink, norm, file.LinkTarget))
				}
				if mapped, ok := renameMap[targetSourceName]; ok {
					out := file
					out.NameInArchive = norm
					out.LinkTarget = mapped
					return emit(out, true)
				}
				// Forward reference under preserve: Tar can encode it, but we
				// must still rewrite the target name once it is known.
				stateMu.Lock()
				pending[targetSourceName] = append(pending[targetSourceName], pendingLink{outName: norm, src: file, preserve: true})
				stateMu.Unlock()
				return nil
			}
		}

		// Regular files under materialization are spooled so later hard links
		// (including forward references) can be resolved within budget.
		if !isDir && ta.Hardlinks == LinkMaterialize {
			se, err := spool.spool(file, norm)
			if err != nil {
				return recordError(err)
			}
			stateMu.Lock()
			retained[sourceName] = se
			stateMu.Unlock()

			if err := emit(se.fileInfo(norm), false); err != nil {
				return err
			}
			if err := flushPending(sourceName, norm, se); err != nil {
				return err
			}
			return nil
		}

		// Plain streaming: directories and regular entries whose Open is
		// borrowed from the sequential source reader. The target consumes the
		// body before this handler returns. Also flush any preserve-mode hard
		// links that were waiting for this entry.
		out := file
		out.NameInArchive = norm
		if !isDir {
			out.Open = limitedOpen(file)
		}
		if err := emit(out, true); err != nil {
			return err
		}
		return flushPending(sourceName, norm, nil)
	}

	extractErr := ta.Source.Extract(childCtx, input, handler)

	// Anything still pending references an entry that never showed up.
	stateMu.Lock()
	var dangling []pendingLink
	for _, pls := range pending {
		dangling = append(dangling, pls...)
	}
	stateMu.Unlock()
	for _, p := range dangling {
		if err := recordError(fmt.Errorf("%w: %q -> %q", ErrDanglingLink, p.outName, p.src.LinkTarget)); err != nil {
			extractErr = errors.Join(extractErr, err)
			break
		}
	}

	// Wait for every pipelined spool job to finish, then signal end-of-input
	// to the target and collect its finalization result.
	inflight.Wait()
	close(jobs)
	targetErr := <-targetDone

	// Surface source and target failures together; target Close/finalize
	// errors are always returned, ContinueOnError notwithstanding.
	var finalErr error
	if extractErr != nil {
		finalErr = errors.Join(finalErr, fmt.Errorf("reading source archive: %w", extractErr))
	}
	if targetErr != nil {
		finalErr = errors.Join(finalErr, fmt.Errorf("writing target archive: %w", targetErr))
	} else if werr := captured.firstError(); werr != nil {
		finalErr = errors.Join(finalErr, fmt.Errorf("writing target archive: %w", werr))
	}
	// Belt-and-braces: the decompressed-byte hard limit must never be silently
	// exceeded even if a target ignores read errors.
	if ta.MaxDecompressedBytes > 0 && spool.bytesRead.Load() > ta.MaxDecompressedBytes {
		finalErr = errors.Join(finalErr, fmt.Errorf("%w: %d bytes read, limit %d",
			ErrDecompressedLimit, spool.bytesRead.Load(), ta.MaxDecompressedBytes))
	}

	result.BytesCopied = spool.bytesRead.Load()
	result.SpoolBytesPeak = spool.peak.Load()
	return result, finalErr
}

// errorCapturingWriter remembers the first error from the underlying writer
// while otherwise behaving as a transparent pass-through.
type errorCapturingWriter struct {
	inner io.Writer
	mu    sync.Mutex
	err   error
}

func (w *errorCapturingWriter) Write(p []byte) (int, error) {
	n, err := w.inner.Write(p)
	if err != nil {
		w.mu.Lock()
		if w.err == nil {
			w.err = err
		}
		w.mu.Unlock()
	}
	return n, err
}

func (w *errorCapturingWriter) firstError() error {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.err
}

// pendingLink remembers a link whose target entry has not been seen yet.
type pendingLink struct {
	outName  string
	src      FileInfo
	preserve bool
}

// spoolManager owns the private temporary directory and the byte accounting
// for content that must outlive the source extract handler.
type spoolManager struct {
	dir               string
	mu                sync.Mutex
	bytesHeld         atomic.Int64
	peak              atomic.Int64
	bytesRead         atomic.Int64
	maxBytes          int64
	decompressedLimit int64
}

func (s *spoolManager) ensureDir() (string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.dir != "" {
		return s.dir, nil
	}
	d, err := os.MkdirTemp("", "archives-transcode-*")
	if err != nil {
		return "", fmt.Errorf("creating private spool: %w", err)
	}
	s.dir = d
	return d, nil
}

func (s *spoolManager) cleanup() {
	s.mu.Lock()
	dir := s.dir
	s.dir = ""
	s.mu.Unlock()
	if dir != "" {
		_ = os.RemoveAll(dir)
	}
}

// spool copies one entry body into the private spool, enforcing both the
// spool budget and the global decompressed-byte budget. The declared size is
// checked up front so an over-budget forward reference fails immediately
// instead of producing an empty placeholder.
func (s *spoolManager) spool(file FileInfo, norm string) (*spoolEntry, error) {
	current := s.bytesHeld.Load()
	if s.maxBytes > 0 && file.Size() > 0 && current+file.Size() > s.maxBytes {
		return nil, fmt.Errorf("%w: entry %q requires %d more bytes but only %d of %d remain",
			ErrSpoolLimit, norm, file.Size(), s.maxBytes-current, s.maxBytes)
	}

	dir, err := s.ensureDir()
	if err != nil {
		return nil, err
	}

	tmp, err := os.CreateTemp(dir, "entry-*")
	if err != nil {
		return nil, fmt.Errorf("creating spool file for %q: %w", norm, err)
	}
	tmpPath := tmp.Name()
	cleanupOnError := func() {
		tmp.Close()
		_ = os.Remove(tmpPath)
	}

	src, err := file.Open()
	if err != nil {
		cleanupOnError()
		return nil, fmt.Errorf("opening source entry %q for spool: %w", norm, err)
	}
	reader := &limitedCountReader{r: src, copied: &s.bytesRead, limit: s.decompressedLimit}
	n, copyErr := io.Copy(tmp, reader)
	closeErr := src.Close()
	if copyErr != nil {
		cleanupOnError()
		return nil, fmt.Errorf("spooling entry %q: %w", norm, copyErr)
	}
	if closeErr != nil {
		cleanupOnError()
		return nil, fmt.Errorf("closing source entry %q after spool: %w", norm, closeErr)
	}
	if err := tmp.Close(); err != nil {
		_ = os.Remove(tmpPath)
		return nil, fmt.Errorf("closing spool file for %q: %w", norm, err)
	}

	if s.maxBytes > 0 && current+n > s.maxBytes {
		_ = os.Remove(tmpPath)
		return nil, fmt.Errorf("%w: entry %q of %d bytes exceeds remaining budget (%d of %d used)",
			ErrSpoolLimit, norm, n, current, s.maxBytes)
	}
	newTotal := s.bytesHeld.Add(n)
	for {
		peak := s.peak.Load()
		if newTotal <= peak || s.peak.CompareAndSwap(peak, newTotal) {
			break
		}
	}

	return &spoolEntry{
		path:    tmpPath,
		name:    norm,
		size:    n,
		mode:    file.Mode(),
		modTime: file.ModTime(),
	}, nil
}

// spoolEntry is a privately retained copy of one source entry's content.
type spoolEntry struct {
	path    string
	name    string
	size    int64
	mode    fs.FileMode
	modTime time.Time
}

// fileInfo builds an independent FileInfo for a spooled entry; its Open does
// not depend on the source stream in any way.
func (s *spoolEntry) fileInfo(nameInArchive string) FileInfo {
	info := staticFileInfo{
		name:    path.Base(nameInArchive),
		size:    s.size,
		mode:    s.mode &^ fs.ModeType, // materialized links become regular files
		modTime: s.modTime,
	}
	return FileInfo{
		FileInfo:      info,
		NameInArchive: nameInArchive,
		Open: func() (fs.File, error) {
			f, err := os.Open(s.path)
			if err != nil {
				return nil, err
			}
			return fileInArchive{ReadCloser: f, info: info}, nil
		},
	}
}

// limitedCountReader counts every byte read and enforces the decompressed
// byte hard limit.
type limitedCountReader struct {
	r      io.Reader
	copied *atomic.Int64
	limit  int64
}

func (l *limitedCountReader) Read(p []byte) (int, error) {
	// Truncate the read at the budget boundary. Delivering the breach on a
	// following read matters because io.CopyN (used by the archive writers)
	// discards a reader error when the full declared Size() arrives in a
	// single read; a subsequent (0, error) read cannot be hidden that way.
	if l.limit > 0 {
		remaining := l.limit - l.copied.Load()
		if remaining <= 0 {
			return 0, fmt.Errorf("%w: %d bytes read, limit %d", ErrDecompressedLimit, l.copied.Load(), l.limit)
		}
		if int64(len(p)) > remaining {
			p = p[:remaining]
		}
	}
	n, err := l.r.Read(p)
	if n > 0 {
		l.copied.Add(int64(n))
	}
	return n, err
}

// limitedFSFile wraps an opened source entry so every byte the target reads
// is counted and bounded.
type limitedFSFile struct {
	fs.File
	reader io.Reader
}

func (l *limitedFSFile) Read(p []byte) (int, error) { return l.reader.Read(p) }

// staticFileInfo is an fs.FileInfo for materialized, spool-backed entries.
type staticFileInfo struct {
	name    string
	size    int64
	mode    fs.FileMode
	modTime time.Time
}

func (s staticFileInfo) Name() string       { return s.name }
func (s staticFileInfo) Size() int64        { return s.size }
func (s staticFileInfo) Mode() fs.FileMode  { return s.mode }
func (s staticFileInfo) ModTime() time.Time { return s.modTime }
func (s staticFileInfo) IsDir() bool        { return s.mode&fs.ModeDir != 0 }
func (s staticFileInfo) Sys() any           { return nil }

// normalizeEntryName canonicalizes an in-archive name and rejects names that
// are empty or escape the archive root.
func normalizeEntryName(name string) (string, error) {
	n := filepath.ToSlash(strings.TrimSpace(name))
	n = path.Clean(n)
	n = strings.TrimPrefix(n, "/")
	if n == "" || n == "." {
		return "", fmt.Errorf("%w: %q", ErrInvalidEntryName, name)
	}
	if n == ".." || strings.HasPrefix(n, ".."+"/") {
		return "", fmt.Errorf("%w: %q escapes the archive root", ErrInvalidEntryName, name)
	}
	return n, nil
}

// uniqueSuffixName returns a name-1, name-2, ... variant not yet present.
func uniqueSuffixName(seen map[string]struct{}, name string) string {
	ext := path.Ext(name)
	stem := strings.TrimSuffix(name, ext)
	for i := 1; ; i++ {
		candidate := fmt.Sprintf("%s-%d%s", stem, i, ext)
		if _, exists := seen[candidate]; !exists {
			seen[candidate] = struct{}{}
			return candidate
		}
	}
}

// resolveInArchiveName resolves a symbolic link target against the directory
// of its containing entry using lexical, in-archive rules only; disk paths are
// never consulted.
func resolveInArchiveName(linkName, target string) string {
	t := filepath.ToSlash(target)
	if path.IsAbs(t) {
		return strings.TrimPrefix(path.Clean(t), "/")
	}
	return path.Clean(path.Join(path.Dir(linkName), t))
}

// rejectUnixOwnership fails entries carrying non-portable Unix owner
// metadata when the target cannot store it (Zip under MetadataReject).
func rejectUnixOwnership(file FileInfo, norm string) error {
	th, ok := file.Header.(*tar.Header)
	if !ok {
		return nil
	}
	var attrs []string
	if th.Uid != 0 {
		attrs = append(attrs, fmt.Sprintf("uid=%d", th.Uid))
	}
	if th.Gid != 0 {
		attrs = append(attrs, fmt.Sprintf("gid=%d", th.Gid))
	}
	if th.Uname != "" {
		attrs = append(attrs, "uname="+th.Uname)
	}
	if th.Gname != "" {
		attrs = append(attrs, "gname="+th.Gname)
	}
	if len(attrs) == 0 {
		return nil
	}
	return fmt.Errorf("%w: entry %q carries non-portable Unix ownership metadata (%s)", ErrUnsupportedMetadata, norm, strings.Join(attrs, ","))
}
