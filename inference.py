from Model import Project
from data import Data
import argparse
from torch.utils.data import Dataset, SubsetRandomSampler



def infer(args):
    data = Data(str(args.test))
    model = Project(*args, **kwargs)
    model.load_state_dict(torch.load(PATH))
    model.eval()
    data_loader = torch.utils.data.DataLoader(
        data, batch_size=len(data), num_workers=24
    )
    for i in range(data_loader):
        results = model(data_loader)
    file = open(args.testcsv, "w")
    file.write("\n".join(results.tolist()))
    file.close()
    print("Done....")
    
    


def main():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--test" help="folder location of the test.csv")
    parser.add_argument("--testcsv" help="file where results are written")
    
    args = parser.parse_args()
    infer(args)

if __name__ == "__main__":
    def main()