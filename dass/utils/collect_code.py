import os
import tarfile


def is_source_file(x):
    if x.isdir() or x.name.endswith(('.py', '.sh', '.yml', '.json', '.txt')) \
            and '.mim' not in x.name and 'jobs/' not in x.name:
        return x
    else:
        return None


def gen_code_archive(out_dir, file='code.tar.gz'):
    archive = os.path.join(out_dir, file)
    os.makedirs(os.path.dirname(archive), exist_ok=True)
    with tarfile.open(archive, mode='w:gz') as tar:
        tar.add('.', filter=is_source_file)
    return archive
