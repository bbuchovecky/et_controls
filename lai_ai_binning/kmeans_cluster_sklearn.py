"""
K-means clustering for a small stack of 2-D images.

This script is designed for an input array with shape (n_images, height, width),
such as (28, 15, 15). It flattens each image into a feature vector, runs
``sklearn.cluster.KMeans``, and returns the cluster label for each image.

Example
-------
    python kmeans_cluster_sklearn.py input.nc --variable lai --n-clusters 4

The input NetCDF file should contain a variable with shape
(n_images, height, width) or any other (n_images, height, width) stack.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr
from sklearn.cluster import KMeans


DEFAULT_OUTPUT_PATH = Path("/glade/work/bbuchovecky/fhist_ppe_analysis/proc/qbin/kmeans/cluster_labels.nc")


class KMeansResult:
    def __init__(self, labels: np.ndarray, centroids: np.ndarray, inertia: float, n_iter: int) -> None:
        self.labels = labels
        self.centroids = centroids
        self.inertia = inertia
        self.n_iter = n_iter


def _prepare_samples(
    samples: np.ndarray,
    nan_policy: str = "mean",
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    If nan_policy='mean', perform per-feature (per-column) mean imputation to fill NaNs. 
    """
    samples = np.asarray(samples, dtype=np.float64)
    finite_mask = np.isfinite(samples)

    if finite_mask.all():
        return samples

    if nan_policy == "raise":
        n_bad = int((~finite_mask).sum())
        raise ValueError(
            f"Input contains {n_bad} non-finite values; choose --nan-policy mean or constant"
        )

    cleaned = samples.copy()

    if nan_policy == "mean":
        counts = finite_mask.sum(axis=0)
        sums = np.where(finite_mask, cleaned, 0.0).sum(axis=0)
        feature_means = np.divide(
            sums,
            counts,
            out=np.full(samples.shape[1], fill_value, dtype=np.float64),
            where=counts > 0,
        )
        bad_entries = ~finite_mask
        cleaned[bad_entries] = feature_means[np.nonzero(bad_entries)[1]]
    elif nan_policy == "constant":
        cleaned[~finite_mask] = fill_value
    else:
        raise ValueError("nan_policy must be one of: raise, mean, constant")

    return cleaned


def _flatten_images(images: np.ndarray) -> np.ndarray:
    if images.ndim != 3:
        raise ValueError(
            f"Expected a 3-D array shaped (n_images, height, width); got {images.shape}"
        )
    n_images = images.shape[0]
    return images.reshape(n_images, -1).astype(np.float64, copy=False)


def kmeans(
    images: np.ndarray,
    n_clusters: int,
    max_iter: int = 300,
    tol: float = 1e-4,
    random_state: int | None = 0,
    nan_policy: str = "mean",
    fill_value: float = 0.0,
) -> KMeansResult:
    """Cluster a stack of images with K-means.

    Parameters
    ----------
    images:
        Array shaped (n_images, height, width).
    n_clusters:
        Number of clusters to form.
    max_iter:
        Maximum K-means iterations.
    tol:
        Convergence threshold on centroid movement.
    random_state:
        Seed used for deterministic centroid initialization.
    nan_policy:
        How to handle non-finite values before clustering. Use "mean" to
        impute each pixel from the finite values in the other images,
        "constant" to replace them with `fill_value`, or "raise" to fail.
    fill_value:
        Replacement value used when nan_policy="constant" or when a feature
        has no finite values under nan_policy="mean".
    """

    samples = _flatten_images(np.asarray(images))
    samples = _prepare_samples(samples, nan_policy=nan_policy, fill_value=fill_value)

    if n_clusters < 1:
        raise ValueError("n_clusters must be at least 1")
    if n_clusters > samples.shape[0]:
        raise ValueError("n_clusters cannot exceed the number of images")

    model = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init="auto",
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
    )
    model.fit(samples)

    return KMeansResult(
        labels=model.labels_.copy(),
        centroids=model.cluster_centers_.copy(),
        inertia=float(model.inertia_),
        n_iter=int(model.n_iter_),
    )


def load_images(
    path: Path,
    variable: str,
    stat: str = "mean",
    transpose: tuple = ("member", "y_bin", "x_bin"),
) -> tuple[np.ndarray, xr.DataArray]:
    with xr.open_dataset(path) as ds:
        if variable not in ds:
            raise ValueError(f"Variable '{variable}' not found in {path}")
        data_xr = ds[variable].sel(stats=stat).transpose(*transpose)
        data = data_xr.load().values
        print("xarray:")
        print(data_xr)
        print(f"numpy: {data.shape}")

    images = np.asarray(data)
    if images.ndim != 3:
        raise ValueError(
            f"Variable '{variable}' must be 3-D (n_images, height, width); got {images.shape}"
        )
    return images, data_xr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster a 3-D NetCDF variable with K-means."
    )
    parser.add_argument(
        "input",
        type=Path,
        help=(
            "Path to a NetCDF file containing a binned variable with dimensions "
            "(stats, y_bin, x_bin, member) computed from `calc_2d_binned_stats.py`"
        ),
    )
    parser.add_argument(
        "--variable",
        type=str,
        required=True,
        help="Variable name in the NetCDF file to cluster",
    )
    parser.add_argument(
        "--n-clusters",
        nargs="+",
        type=int,
        help="Number of clusters to form"
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=300,
        help="Maximum number of K-means iterations"
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-5,
        help="Convergence tolerance on centroid movement"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for centroid initialization"
    )
    parser.add_argument(
        "--nan-policy",
        type=str,
        default="mean",
        choices=("mean", "constant", "raise"),
        help=(
            "How to handle non-finite values before clustering: "
            "mean, constant, or raise"
        ),
    )
    parser.add_argument(
        "--nan-fill-value",
        type=float,
        default=0.0,
        help="Replacement value used when --nan-policy=constant",
    )
    parser.add_argument(
        "--abs",
        action="store_true",
        help="Use the absolute value of the images."
    )
    parser.add_argument(
        "--labels-out",
        type=Path,
        default=None,
        help=(
            "Path to save cluster labels as a DataArray. "
            "Default is <input_path>.cluster_labels.nc"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run, do not save cluster labels"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images, data_xr = load_images(args.input, args.variable)

    abs_tag = ""
    if args.abs:
        images = abs(images)
        abs_tag = "abs"

    for nc in args.n_clusters:

        result = kmeans(
            images=images,
            n_clusters=nc,
            max_iter=args.max_iter,
            tol=args.tol,
            random_state=args.seed,
            nan_policy=args.nan_policy,
            fill_value=args.nan_fill_value,
        )

        print(f"Input shape: {images.shape}")
        print(f"Variable: {args.variable}")
        print(f"Cluster labels: {result.labels.tolist()}")
        print(f"Inertia: {result.inertia:.6f}")
        print(f"Iterations: {result.n_iter}")

        sample_dim = data_xr.dims[0]
        labels_da = xr.DataArray(
            result.labels,
            dims=(sample_dim,),
            coords={sample_dim: data_xr.coords[sample_dim]},
            name=f"{args.variable}_cluster",
            attrs={
                "source_variable": args.variable,
                "n_clusters": int(nc),
                "inertia": float(result.inertia),
                "n_iter": int(result.n_iter),
                "random_seed": int(args.seed),
                "nan_policy": args.nan_policy,
                "nan_fill_value": float(args.nan_fill_value),
                "source_file": str(args.input),
            },
        )
        print("Cluster labels:")
        print(labels_da)

    
        if args.labels_out is None:
            fout = f"{args.input.with_suffix('')}.{nc}_{abs_tag}cluster_labels.nc"
        else:
            fout = args.labels_out
            fout.parent.mkdir(parents=True, exist_ok=True)
        print(f"Save labels to: {fout}")

        if not args.dry_run:
            labels_da.to_netcdf(fout)
            print(f"Labels saved")


        for cluster_id in range(nc):
            # members = np.where(result.labels == cluster_id)[0]
            members = labels_da.where(labels_da == cluster_id, drop=True)["member"].values
            print(f"Cluster {cluster_id}: images {members.tolist()}")

if __name__ == "__main__":
    main()
