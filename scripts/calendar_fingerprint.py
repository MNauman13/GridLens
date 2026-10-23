"""Require identical cyclic calendar values across Linux and Windows CI."""
import argparse
import json
import os
from gridlens.data.dataset import cyclic
from gridlens.data.store import content_id

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--github-output",action="store_true")
    args=parser.parse_args()
    digest=content_id({str(n):[cyclic(i,n) for i in range(n)] for n in (7,48,17520,17568)})
    expected=os.environ.get("EXPECTED_CALENDAR_SHA256")
    if expected and digest!=expected:
        raise SystemExit("Calendar arithmetic differs across operating systems")
    if args.github_output:
        with open(os.environ["GITHUB_OUTPUT"],"a",encoding="utf-8") as stream:
            stream.write("digest="+digest+"\n")
    print(json.dumps({"calendar_sha256":digest,"compared_to_linux":bool(expected)}))
if __name__=="__main__":
    main()
