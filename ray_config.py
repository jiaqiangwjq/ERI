import argparse
parser = argparse.ArgumentParser(description='TEMMA')

parser.add_argument('--mask_a_length', type=str, default='50,50')
parser.add_argument('--mask_b_length', type=str, default='10,10')
parser.add_argument('--block_num', type=int, default=4)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--dropout_mmatten', type=float, default=0.5)
parser.add_argument('--dropout_mtatten', type=float, default=0.2)
parser.add_argument('--dropout_ff', type=float, default=0.2)
parser.add_argument('--dropout_subconnect', type=float, default=0.2)
parser.add_argument('--dropout_position', type=float, default=0.2)
parser.add_argument('--dropout_embed', type=float, default=0.2)
parser.add_argument('--dropout_fc', type=float, default=0.2)
parser.add_argument('--h', type=int, default=4)
parser.add_argument('--h_mma', type=int, default=4)
parser.add_argument('--d_model', type=int, default=128)
parser.add_argument('--d_ff', type=int, default=256)
parser.add_argument('--modal_num', type=int, default=2)
parser.add_argument('--embed', type=str, default='temporal')
parser.add_argument('--levels', type=int, default=5)
parser.add_argument('--ksize', type=int, default=3)
parser.add_argument('--ntarget', type=int, default=2)

parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--tensorboard_log', type=str, default="log_1")
parser.add_argument('--save_model_dir', type=str, default="dir_1")
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--use_cuda', type=bool, default=True)
parser.add_argument('--feature_name', type=str, default="enet+fabnet")

opts = parser.parse_args()



opts.modal_num = 1
opts.mask_a_length = '75'
opts.mask_b_length = '75'
opts.ntarget = 7