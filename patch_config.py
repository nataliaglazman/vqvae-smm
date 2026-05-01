with open('config.py', 'r') as f:
    code = f.read()

code = code.replace(
    'parser.add_argument("--crop-margin", type=int, default=0, help="Voxels to crop from each edge")',
    'parser.add_argument("--crop-margin", type=int, default=0, help="Voxels to crop from each edge")\n    parser.add_argument("--spatial-size", type=int, nargs="+", default=None, help="Explicit spatial size (depth height width) in voxels. Overrides size derived from spacing and crop-margin.")'
)

with open('config.py', 'w') as f:
    f.write(code)
