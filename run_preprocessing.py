import os
from configs.args import get_args
from data.preprocessor import Preprocessor
from utils.logger import logger

def main():
    args = get_args()
    
    # 1. Automatic Parameter Configuration based on Dataset
    freq = args.frequency
    stride = args.stride
    
    if args.dataset == "gazecom":
        freq = freq if freq is not None else 250
        stride = stride if stride is not None else 10
    elif args.dataset == "hmr":
        freq = freq if freq is not None else 200
        stride = stride if stride is not None else 8
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    final_out_path = os.path.join(args.processed_data_path, args.dataset)
    
    logger.info(f" Preprocessing Setup for {args.dataset.upper()}:")
    logger.info(f"   - Frequency: {freq} Hz")
    logger.info(f"   - Stride: {stride}")
    logger.info(f"   - Window Length: {args.window_length}")
    logger.info(f"   - Offset: {args.offset}")
    logger.info(f"   - Raw Data Path: {args.raw_data_path}")
    
    if not os.path.exists(args.raw_data_path):
        logger.error(f" Raw data path '{args.raw_data_path}' does not exist! Please check your dataset folder.")
        return

    # 2. Instantiate Preprocessor
    pprep = Preprocessor(
        stride=stride, 
        frequency=freq, 
        window_length=args.window_length, 
        offset=args.offset
    )
    
    # 3. Start Extraction
    logger.info(f" Starting feature extraction...")
    pprep.process_folder(base_path=args.raw_data_path, out_path=final_out_path)
    
    actual_folder_name = f"{args.dataset}_s{stride}_f{freq}_w{args.window_length}_o{args.offset}"
    final_folder = os.path.join(args.processed_data_path, actual_folder_name)
    
    logger.info(" Feature extraction completed successfully!")
    logger.info(f" You can now run training pointing to: {final_folder}")

if __name__ == "__main__":
    main()