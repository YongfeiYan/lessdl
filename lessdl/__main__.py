from lessdl import parse_args, run_main

if __name__ == '__main__':
    args = parse_args()
    run_main(args, args.evaluate_best_ckpt, args.evaluate_only)
