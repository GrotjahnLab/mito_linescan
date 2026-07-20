from setuptools import setup, find_packages

setup(
    name="mito-linescan",
    version="0.1",
    description="Mitochondrial protein line scanner tools",
    author="Hamid Rahmani, Michaela Horger",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "mito_protein_localization=mito_linescan.mito_protein_localization:main",
            "mito_mask=mito_linescan.mito_mask:main",
            "mito_mask_refine=mito_linescan.mito_mask_refine:main",
            "mito_protein_line_scanner=mito_linescan.mito_protein_line_scanner:main",
            "mito_protein_omm_normal_scanner=mito_linescan.mito_protein_omm_normal_scanner:main",
            "mask_thickness=mito_linescan.mask_thickness_util:main",
        ],
    },
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "pandas",
        "scikit-image",
        "networkx",
        "tqdm",
        "click",
        "tifffile",
        "sknw",
        "pyyaml",
        "mrcfile",
    ],
)
