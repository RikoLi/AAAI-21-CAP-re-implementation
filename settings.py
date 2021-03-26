import argparse

class Settings:
    conf = None
    gpu_ids = None
    is_continue = None
    use_pretrained = None
    debug = False

    @classmethod
    def init(cls):
        parser = argparse.ArgumentParser(usage='python train.py [--is_continue] --conf <path> --gpu_ids <0,1,...>')
        parser.add_argument('--conf', type=str, help='Config file path.', required=True)
        parser.add_argument('--gpu_ids', type=str, help='Available GPU IDs.', required=True)
        parser.add_argument('--is_continue', help='Continue training from the latest checkpoint. It should only be used when training is interrupted before expected epoches.', action='store_true')
        parser.add_argument('--use_pretrained', help='Use pretrain checkpoints from the config file.', action='store_true')
        parser.add_argument('--debug', help='Start with debug mode.', action='store_true')
        args = parser.parse_args()
        cls.conf = args.conf
        cls.gpu_ids = args.gpu_ids
        cls.is_continue = args.is_continue
        cls.use_pretrained = args.use_pretrained
        cls.debug = args.debug

        if cls.is_continue and cls.use_pretrained:
            raise ValueError('--is_continue and --use_pretrained cannot be given together!')

if __name__ == '__main__':
    Settings.init()
    print(Settings.is_continue)