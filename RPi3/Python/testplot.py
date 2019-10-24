import numpy as np
import matplotlib.pyplot as plt
import time




if __name__ == "__main__":
    
    plt.axis([0, 100, 0, 1])
    graph = list()
    counter = 0

    try: 
                
        while (1): 

            y = np.random.random()

            graph.append(y)

            plt.scatter(counter, y)

            plt.pause(0.01)

            counter += 1

            if (counter == 100):
                graph.pop(0)
                counter -= 1

    except KeyboardInterrupt:
        plt.show()
        print('KeyboardInterrupt')