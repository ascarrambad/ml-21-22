from tensorflow.keras.models import load_model

from src.utils import load_cifar10

if __name__ == '__main__':

    # Load the test CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = load_cifar10(num_classes=3)


    # Preprocessing

    # ...


    # Load the trained models (one or more depending on task and bonus)
    # for example
    model_task1 = load_model('./nn_task1.h5')


    # Predict on the given samples
    # for example
    y_pred_task1 = model_task1.predict(x_test)


    # Evaluate the missclassification error on the test set
    # for example
    assert y_test.shape == y_pred_task1.shape
    acc1 = (y_test == y_pred_task1).mean()
    print("Accuracy model task 1:", acc1)
