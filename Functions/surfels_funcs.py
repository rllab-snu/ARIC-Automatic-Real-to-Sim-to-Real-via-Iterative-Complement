import sys
import torch
import pymeshlab
sys.path.append('./References/Gaussian_Surfels')
from argparse import ArgumentParser
from References.Gaussian_Surfels.arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from References.Gaussian_Surfels.train_surfels import training
from References.Gaussian_Surfels.render_surfels import render_sets, safe_state, generate_mesh_from_pcd


def optimize_gaussian_surfels(source_path, output_path, object_index):
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[5_000, 10_000, 15_000, 20_000, 25_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[5_000, 15_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    # args = parser.parse_args(sys.argv[1:])
    args, unknown = parser.parse_known_args(sys.argv[1:])


    args.save_iterations.append(args.iterations)
    args.source_path = source_path
    args.model_path = output_path
    # print("Optimizing " + args.model_path)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(
        lp.extract(args), op.extract(args), pp.extract(args),
        object_index,
        args.test_iterations, args.save_iterations, args.checkpoint_iterations,
        args.start_checkpoint, args.debug_from
    )

def render_gaussian_surfels(output_path):
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--depth", default=10, type=int)
    args = get_combined_args(parser, output_path)

    args.skip_train = False
    args.skip_test = False
    args.quiet = False
    args.img = True

    safe_state(args.quiet)
    sampled_points = render_sets(
        model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.img, args.depth
    )
    return sampled_points


def convert_mesh(ply_path, ply_name):
    input_ply = "{}/{}_pruned.ply".format(ply_path, ply_name)
    output_obj = "{}/{}_textured_mesh.obj".format(ply_path, ply_name)

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(input_ply)
    ms.compute_texcoord_parametrization_triangle_trivial_per_wedge()
    ms.compute_texmap_from_color(textname="{}_mesh_text".format(ply_name))
    ms.save_current_mesh(output_obj)

def generate_mesh(object_points, object_path, object_name=""):
    generate_mesh_from_pcd(object_points, object_path, object_name)
 