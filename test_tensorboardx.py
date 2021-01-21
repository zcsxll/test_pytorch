from tensorboardX import SummaryWriter
import numpy as np

def run():
    logger = SummaryWriter('./log')
    for i in range(200, 400):
        logger.add_scalar('STOI/SER 0', np.random.randn(), i)
    for i in range(200, 400):
        logger.add_scalar('STOI/SER 5', np.random.randn(), i)
    for i in range(200, 400):
        logger.add_scalar('STOI/SER 10', np.random.randn(), i)
    for i in range(0, 300):
        logger.add_scalar('PESQ/SER 0', np.random.randn(), i)

if __name__ == '__main__':
    run()
