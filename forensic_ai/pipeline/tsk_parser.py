import pytsk3

def parse_mft(image_path):
    img = pytsk3.Img_Info(image_path)
    fs = pytsk3.FS_Info(img)

    records = []

    def walk(dir_obj):
        for entry in dir_obj:
            try:
                if not entry.info.meta:
                    continue

                meta = entry.info.meta
                name = entry.info.name.name.decode(errors="ignore")

                records.append({
                    "inode": meta.addr,
                    "size": meta.size,
                    "timestamp": meta.mtime or 0
                })

                if meta.type == pytsk3.TSK_FS_META_TYPE_DIR:
                    walk(entry.as_directory())

            except Exception:
                continue

    walk(fs.open_dir("/"))
    return records