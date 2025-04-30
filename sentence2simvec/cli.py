import argparse, pathlib, os, numpy as np
from .core import similarity_score

def _main() -> None:
    ap = argparse.ArgumentParser(prog="sentence2simvec")
    ap.add_argument("text1")
    ap.add_argument("text2")
    ap.add_argument("--w-ngram", type=float, default=0.3)
    ap.add_argument("--w-embed", type=float, default=0.7)
    ap.add_argument("--save-vecs", metavar="DIR", help="save vec1.npy & vec2.npy here")
    args = ap.parse_args()

    score, detail, *vecs = similarity_score(
        args.text1, args.text2,
        w_ngram=args.w_ngram, w_embed=args.w_embed,
        return_breakdown=True, return_vectors=bool(args.save_vecs)
    )
    print(f"Similarity: {score:.4f}")
    print(f"  • n-gram   = {detail['jaccard']:.4f}")
    print(f"  • cosine   = {detail['cosine']:.4f}")

    if args.save_vecs:
        d = pathlib.Path(args.save_vecs)
        d.mkdir(parents=True, exist_ok=True)
        np.save(os.fspath(d / "vec1.npy"), vecs[0])
        np.save(os.fspath(d / "vec2.npy"), vecs[1])
        print(f"Vectors saved under {d}/vec*.npy")

if __name__ == "__main__":
    _main()
