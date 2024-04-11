#python main.py -log -data cifar100 -gr 800 -algo FedCAG -m cnn -nc 20 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.3 -lbs 32 --noniid --balance --alpha_dirich 1 -nb 100
#python main.py -log -data cifar100 -gr 800 -algo FedCAG -m cnn -nc 40 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.3 -lbs 32 --noniid --balance --alpha_dirich 1 -did 1
#python main.py -log -data cifar100 -gr 800 -algo FedCAG -m cnn -nc 60 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.3 -lbs 32 --noniid --balance --alpha_dirich 1 -did 1
#
#python main.py -log -data emnist -gr 100 -algo FedCagRod -m cnn -nc 20 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.3 -lbs 32 --noniid --balance --alpha_dirich 1 -did 1
#python main.py -log -data emnist -gr 200 -algo FedCagRod -m cnn -nc 40 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.3 -lbs 32 --noniid --balance --alpha_dirich 1 -did 1
#python main.py -log -data emnist -gr 400 -algo FedCagRod -m cnn -nc 60 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.3 -lbs 32 --noniid --balance --alpha_dirich 1 -did 1

python main.py -data cifar100 -gr 800 -algo FedCAG -m resnet10 -mstr resnet10 -nc 60 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.9 -lbs 32 -c 0.9 --noniid --balance --alpha_dirich 0.1 -nb 100
python main.py -data cifar100 -gr 800 -algo FedCAG -m resnet10 -mstr resnet10 -nc 60 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.9 -lbs 32 -c 0.1 --noniid --balance --alpha_dirich 0.1 -nb 100
python main.py -data cifar100 -gr 800 -algo FedCAG -m resnet10 -mstr resnet10 -nc 60 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.3 -lbs 32 -c 0.9 --noniid --balance --alpha_dirich 0.1 -nb 100
python main.py -log -data mnist -gr 800 -algo FedCagRod -m cnn -nc 60 -ls 5 -car 100 -calr 25 -mmt 0.5 -ss 30 -gam 0.3 -lbs 32 --noniid --balance --alpha_dirich 0.1 