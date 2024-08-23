from Transform import *
from sys import argv

if __name__ == "__main__":

    video = argv

    n_frames = 10
    batch_size = 1

    model = tf.keras.models.load_model('video_model.keras',
                                       custom_objects={'Conv2Plus1D': Conv2Plus1D,
                'ResidualMain': ResidualMain,
                 'Project': Project,
                  'add_residual_block': add_residual_block,
                  'ResizeVideo': ResizeVideo})

    output_signature = (tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float32),
                        tf.TensorSpec(shape=(), dtype=tf.int16))

    test_video = tf.data.Dataset.from_generator(FrameGenerator(video, n_frames),
                                                output_signature=output_signature)

    test_video = test_video.batch(batch_size)

    def prediction(video_to_predict):

        predicted = model.predict(video_to_predict)
        predicted = tf.argmax(predicted, axis=1)

        predicted = predicted.numpy()[0]

        return predicted

    prediction(test_video)






