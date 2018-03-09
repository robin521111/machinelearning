import utils
import matplotlib.pyplot as plt

# visualize the model prediction results
def visualize_model_label(RGB_history):
    print(RGB_history.history['loss'])
    plt.figure(figsize=(9,6))
    plt.plot(RGB_history.history['loss'])
    plt.title('model loss')
    plt.ylabel('Loss', fontsize=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.legend(['train Loss'], loc='upper right')
    plt.grid()
    #plt.savefig(img_dir + "/nvidia_loss_compare.png", dpi=300)
    plt.show()