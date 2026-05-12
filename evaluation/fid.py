import subprocess
import sys


def compute_fid(
    path_real,
    path_fake,
    device="cuda:0",
    batch_size=None,
    num_workers=None,
    timeout=600,
):
    """Compute FID between two directories of images using pytorch-fid.

    Returns FID score as float, or None if computation fails.
    """
    try:
        cmd = [
            sys.executable, "-m", "pytorch_fid",
            path_real, path_fake,
            "--device", device,
        ]
        if batch_size is not None:
            cmd.extend(["--batch-size", str(batch_size)])
        if num_workers is not None:
            cmd.extend(["--num-workers", str(num_workers)])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        # Parse FID from output: "FID:  XX.XX"
        for line in result.stdout.strip().split("\n"):
            if "FID" in line:
                fid = float(line.split(":")[-1].strip())
                return fid
        print(f"pytorch-fid stdout: {result.stdout}")
        print(f"pytorch-fid stderr: {result.stderr}")
        return None
    except Exception as e:
        print(f"FID computation failed: {e}")
        return None
