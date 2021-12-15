import argparse
import pprint


def parse_options(return_parser=False):
    parser = argparse.ArgumentParser(description='image colorization')
    global_group = parser.add_argument_group('global')
    global_group.add_argument('--train-path', type=str, default = 'data/train/',
                              help='path to the training dataset')
    global_group.add_argument('--test-path', type=str, default = 'data/test/',
                              help='path to the testing dataset')
    global_group.add_argument('--batch-size', type=int, default=16,
                              help='batch size')
    global_group.add_argument('--lr',type=float,default=0.0001,
                              help='the learning rate.')
    global_group.add_argument('--optimizer', type=str, default = 'adam',
                              help='optimizer to use')
    global_group.add_argument('--epochs', type=int, default=200,
                              help='training epoch')
    global_group.add_argument('--valid-every', type=int, default=1,
                              help='training epoch')
    global_group.add_argument('--save-every', type =int, default=1,
                                help='the frequency of saving model')
    global_group.add_argument('--model-path',type=str,default='model/',
                              help='The path to the saved model')

    preprocess_group = parser.add_argument_group('preprocess')

    preprocess_group.add_argument('--transforms', type=str, nargs='*', 
                            default=['rotation','rotation','crop','crop','flip'],
                            help='The data augmentation methods.')
    preprocess_group.add_argument('--img-path', type=str, default = 'landscape_images',
                              help='path to the images')
    preprocess_group.add_argument('--train-test-ratio', type=float, default = 0.8,
                              help='the ratio of training images over all images')
    
    # Parse and run
    if return_parser:
        return parser
    else:
        return argparse_to_str(parser)


def argparse_to_str(parser):
    """Convert parser to string representation for Tensorboard logging.
    Args:
        parser (argparse.parser): CLI parser
    """

    args = parser.parse_args()

    args_dict = {}
    for group in parser._action_groups:
        group_dict = {a.dest:getattr(args, a.dest, None) for a in group._group_actions}
        args_dict[group.title] = vars(argparse.Namespace(**group_dict))

    pp = pprint.PrettyPrinter(indent=2)
    args_str = pp.pformat(args_dict)
    args_str = f'```{args_str}```'

    return args, args_str