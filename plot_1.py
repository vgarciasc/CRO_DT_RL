import matplotlib.pyplot as plt
import numpy as np
import pdb
from cycler import cycler


if __name__ == "__main__":
    plt.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')

    oct_avg_d2 = [0.945, 0.737, 0.901, 0.671, 1.000, 1.000, 0.755, 0.905, 0.712, 0.294, 0, 0, 0, 0]
    cart_avg_d2 = [0.938, 0.778, 0.894, 0.647, 0.867, 0.990, 0.761, 0.913, 0.694, 0.250, 0.551, 0.502, 0.552, 0.511]
    c45_avg_d2 = [0.931, 0.791, 0.887, 0.353, 0.973, 0.988, 0.754, 0.899, 0.664, 0.132, 0.230, 0.382, 0.00, 0.00]
    gosdt_avg_d2 = [0.953, 0.776, 0.911, 0.691, 0.997, 0.989, 0.755, 0.919, 0.695, 0.384, 0.621, 0.505, 0.509, 0.467]
    tao_avg_d2 = [0.942, 0.778, 0.898, 0.664, 0.912, 0.990, 0.756, 0.906, 0.687, 0.315, 0.652, 0.510, 0.550, 0.515]
    crodt_avg_d2 = [0.939, 0.776, 0.885, 0.692, 1.000, 1.000, 0.759, 0.915, 0.728, 0.344, 0, 0, 0, 0]
    crodt_cs_avg_d2 = [0.956, 0.775, 0.904, 0.693, 1.000, 0.998, 0.758, 0.913, 0.700, 0.380, 0.570, 0.536, 0.511, 0.482]

    oct_avg_d3 = [0.953, 0.774, 0.896, 0.689, 1.000, 1.000, 0.770, 0.914, 0.696, 0.416, 0, 0, 0, 0]
    cart_avg_d3 = [0.947, 0.778, 0.918, 0.688, 1.000, 0.990, 0.765, 0.916, 0.701, 0.346, 0.766, 0.528, 0.553, 0.519]
    c45_avg_d3 = [0.946, 0.858, 0.915, 0.355, 0.985, 0.988, 0.751, 0.889, 0.663, 0.132, 0.231, 0.377, 0.32, 0.33]
    gosdt_avg_d3 = [0.952, 0.809, 0.965, 0.728, 0.997, 0.989, 0.778, 0.919, 0.712, 0.599, 0.775, 0.530, 0.511, 0.469]
    tao_avg_d3 = [0.947, 0.793, 0.949, 0.717, 0.997, 0.990, 0.769, 0.905, 0.709, 0.499, 0.782, 0.536, 0.561, 0.520]
    crodt_avg_d3 = [0.941, 0.805, 0.915, 0.726, 1.000, 1.000, 0.756, 0.905, 0.714, 0.500, 0, 0, 0, 0]
    crodt_cs_avg_d3 = [0.950, 0.801, 0.929, 0.728, 1.000, 0.998, 0.759, 0.909, 0.711, 0.544, 0.663, 0.569, 0.517, 0.490]
    
    oct_avg_d4 = [0.953, 0.788, 0.907, 0.716, 1.000, 1.000, 0.770, 0.914, 0.696, 0.547, 0, 0, 0, 0]
    cart_avg_d4 = [0.947, 0.843, 0.936, 0.749, 1.000, 0.990, 0.771, 0.918, 0.706, 0.532, 0.805, 0.540, 0.559, 0.520]
    c45_avg_d4 = [0.944, 0.879, 0.971, 0.364, 0.985, 0.988, 0.748, 0.883, 0.664, 0.134, 0.231, 0.3770, 0.320, 0.33]
    gosdt_avg_d4 = [0.953, 0.864, 0.978, 0.783, 0.997, 0.989, 0.773, 0.923, 0.716, 0.636, 0.651, 0.537, 0.508, 0.471]
    tao_avg_d4 = [0.947, 0.845, 0.966, 0.774, 0.997, 0.990, 0.767, 0.901, 0.709, 0.646, 0.832, 0.558, 0.569, 0.523]
    crodt_avg_d4 = [0.943, 0.855, 0.945, 0.765, 1.000, 1.000, 0.758, 0.898, 0.726, 0.600, 0, 0, 0, 0]
    crodt_cs_avg_d4 = [0.951, 0.863, 0.955, 0.780, 1.000, 0.997, 0.764, 0.908, 0.715, 0.649, 0.712, 0.595, 0.518, 0.495]

    datasets = ["Breast cancer", "Car evaluation", "Banknote authentication", "Balance scale", "Acute inflammations 1", "Acute inflammations 2", "Blood transfusion", "Climate model crashes", "Connectionist bench sonar", "Optical recognition", "Drybeans", "Avila bible", "Wine quality red", "Wine quality white"]
    
    oct_data = [oct_avg_d2, oct_avg_d3, oct_avg_d4]
    cart_data = [cart_avg_d2, cart_avg_d3, cart_avg_d4]
    c45_data = [c45_avg_d2, c45_avg_d3, c45_avg_d4]
    gosdt_data = [gosdt_avg_d2, gosdt_avg_d3, gosdt_avg_d4]
    tao_data = [tao_avg_d2, tao_avg_d3, tao_avg_d4]
    crodt_data = [crodt_avg_d2, crodt_avg_d3, crodt_avg_d4]
    crodt_cs_data = [crodt_cs_avg_d2, crodt_cs_avg_d3, crodt_cs_avg_d4]

    foobar = lambda res, i : np.array([(x, res[x][i]) for x in range(3)])

    fig, axs = plt.subplots(3, 5, sharex=True)
    axs[-1, -1].axis('off')
    for i, dataset in enumerate(datasets):
        ax = axs[i//5, i%5]
        ax.plot(foobar(oct_data, i)[:, 0], foobar(oct_data, i)[:, 1], marker='^', label="OCT")
        ax.plot(foobar(cart_data, i)[:, 0], foobar(cart_data, i)[:, 1], marker='^', label="CART")
        ax.plot(foobar(gosdt_data, i)[:, 0], foobar(gosdt_data, i)[:, 1], marker='^', label="GOSDT")
        ax.plot(foobar(c45_data, i)[:, 0], foobar(c45_data, i)[:, 1], marker='^', label="C4.5")
        ax.plot(foobar(tao_data, i)[:, 0], foobar(tao_data, i)[:, 1], marker='^', label="TAO")
        ax.plot(foobar(crodt_data, i)[:, 0], foobar(crodt_data, i)[:, 1], marker='^', label="CRO-DT")
        ax.plot(foobar(crodt_cs_data, i)[:, 0], foobar(crodt_cs_data, i)[:, 1], marker='^', label="CRO-DT (CS)")
        ax.set_xticks([0, 1, 2], ["2", "3", "4"])
        ax.set_title(dataset)
        
        if i >= 2*5:
            ax.set_xlabel("Depth")
        
        if i == 13:
            ax.legend(bbox_to_anchor=(1.05, 0.5), loc="center left")

    axs[1, 0].set_ylabel("Average out-of-sample accuracy")
    plt.subplots_adjust(left=0.06, bottom=0.07, right=0.983, top=0.957, wspace=0.25, hspace=0.2)
    plt.show()