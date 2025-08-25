import pickle

def print_pickle_file(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
        print(data)

if __name__ == "__main__":
    print_pickle_file("/home/magraz/research/src/training_stats.pkl")