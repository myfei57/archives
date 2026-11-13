//go:build windows

package archives

import "io/fs"

// canonicalInodeKey identifies a physical file for hard-link grouping.
// Windows currently exposes no portable inode information through
// hardlinkKey, so the default detector never groups entries here.
type canonicalInodeKey struct {
	dev uint64
	ino uint64
}

func defaultHardlinkKey(_ fs.FileInfo) (canonicalInodeKey, bool) {
	return canonicalInodeKey{}, false
}
