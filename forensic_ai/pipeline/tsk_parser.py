import pytsk3
import datetime

def parse_mft(image_path):
    img = pytsk3.Img_Info(image_path)
    fs = pytsk3.FS_Info(img)

    records = []

    directory = fs.open_dir(path="/")

    def walk_directory(directory, parent_path=""):
        for entry in directory:
            try:
                if not hasattr(entry, "info") or not entry.info.meta:
                    continue

                meta = entry.info.meta
                name = entry.info.name.name.decode(errors="ignore")

                record = {
                    "file_name": name,
                    "inode": meta.addr,
                    "parent_path": parent_path,
                    "crtime": meta.crtime,
                    "mtime": meta.mtime,
                    "atime": meta.atime,
                    "ctime": meta.ctime,
                    "size": meta.size
                }

                records.append(record)

                # Recurse into directories
                if entry.info.meta.type == pytsk3.TSK_FS_META_TYPE_DIR:
                    subdir = entry.as_directory()
                    walk_directory(subdir, parent_path + "/" + name)

            except Exception:
                continue

    walk_directory(directory)

    return records