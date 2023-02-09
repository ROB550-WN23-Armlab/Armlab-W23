import numpy as np
import numpy.ma as ma


def main():
    check = np.zeros((3,3))

    print(check)

    mask = np.ones_like(check)
    mask[1:2,1:2] -=1

    print(mask)

    thesevals = ma.masked_array(check,mask)

    print(thesevals)




if __name__ == '__main__':
    main()