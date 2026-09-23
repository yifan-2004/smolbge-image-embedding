"""python example.py --images photo1.jpg photo2.jpg --query 'a dog outdoors'"""
import argparse
import json

from model import ImageEmbeddingModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=".")
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--cache-dir", help="Optional existing Hugging Face model cache")
    args = parser.parse_args()
    model = ImageEmbeddingModel.from_pretrained(args.model, device=args.device, cache_dir=args.cache_dir)
    images = model.encode_images(args.images)
    query = model.encode_text(args.query)
    scores = images @ query
    ranks = scores.argsort()[::-1]
    print(json.dumps([{"image": args.images[i], "cosine": float(scores[i])} for i in ranks], indent=2))


if __name__ == "__main__":
    main()
