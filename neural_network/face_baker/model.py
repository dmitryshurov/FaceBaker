from tensorflow import keras


class ModelBuilder:
    def __init__(self,
                 dense_layers_width=256,
                 num_dense_layers=8,
                 leaky_relu_alpha=0.3,        # Since it's not clear from the paper, the default is 0.3
                 skip_connections_type='add'  # Since it's not clear from the paper, this can be 'add' or 'concat'
                 ):
        self.dense_layers_width = dense_layers_width
        self.num_dense_layers = num_dense_layers
        self.leaky_relu_alpha = leaky_relu_alpha
        self.skip_connections_type = skip_connections_type
        self.model = None

    @staticmethod
    def dense_layer_name(layer_num):
        return f'Dense_{layer_num}'

    def create_model(
            self,
            num_rig_control_variables=100,
            num_output_mesh_vertices=1000,
            num_pca_components=50):

        # Add the flat input layer
        input_layer = keras.Input(shape=(num_rig_control_variables,), name="Input_Rig_Control_Variables")

        # Add the first Dense layer
        dense_layer = keras.layers.Dense(self.dense_layers_width, name=self.dense_layer_name(1))(input_layer)

        # We don't need skip connections after the first Dense layer
        skip_layer = None

        # Add the rest 7 layer blocks
        for layer_num in range(2, self.num_dense_layers + 1):
            dense_layer, skip_layer = self._add_dense_layer_with_skip_connections(layer_num, dense_layer, skip_layer)

        # Add missing Leaky ReLU
        leaky_relu = keras.layers.LeakyReLU(alpha=self.leaky_relu_alpha)(skip_layer)

        # Add PCA Dense layer
        pca_layer = keras.layers.Dense(num_pca_components, name="PCA")(leaky_relu)

        # Add output Dense layer
        output_layer = keras.layers.Dense(num_output_mesh_vertices * 3, name="Output_Mesh_Coordinates")(pca_layer)

        # Create the model
        self.model = keras.Model(inputs=input_layer, outputs=output_layer, name="FaceBaker")

        return self.model

    def _add_dense_layer_with_skip_connections(
            self,
            layer_num,
            dense_layer_prev,
            skip_layer_prev):

        """
        Add a Dense - Leaky ReLU - Skip Connection block as in Figure 2
        """

        # After the first Dense layer we don't have a subsequent Add layer, so use that Dense layer instead
        if skip_layer_prev is None:
            skip_layer_prev = dense_layer_prev

        leaky_relu = keras.layers.LeakyReLU(alpha=self.leaky_relu_alpha)(skip_layer_prev)
        dense_layer_new = keras.layers.Dense(self.dense_layers_width, name=self.dense_layer_name(layer_num))(leaky_relu)
        skip_layer_new = keras.layers.Add()([dense_layer_prev, dense_layer_new])

        return dense_layer_new, skip_layer_new

    def plot_model_to_file(self, file_path):
        keras.utils.plot_model(self.model, file_path, show_shapes=True)
