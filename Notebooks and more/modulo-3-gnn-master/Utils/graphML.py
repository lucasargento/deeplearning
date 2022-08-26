# 2018/11/01~
# Fernando Gama, fgama@seas.upenn.edu.
"""
graphML.py Module for basic GSP and graph machine learning functions.

Functionals

graphConv: Applies a graph convolutional filter

Filtering Layers (nn.Module)

GraphConv: Creates a graph convolutional layer
"""

import math
import torch
import torch.nn as nn

zeroTolerance = 1e-9 # Values below this number are considered zero.
infiniteNumber = 1e12 # infinity equals this number

def graphConv(input: torch.Tensor,
              graph: torch.Tensor,
              weight: torch.Tensor,
              bias: torch.Tensor = None):
    r"""Applies a graph convolution (polynomial/FIR implementation)
    to an input graph signal.

    See :ref:`GraphConv` for details.

    Args:
        input (Tensor): input graph signal of shape
        :math:`(\text{minibatch}, \text{trajectory}, \text{in\_features}, \text{nodes})`
        graph (Tensor): input graph matrix description of shape
        :math:`(\text{minibatch}, \text{trajectory}, \text{edge\_features}, \text{nodes}, \text{nodes})`
        weight (Tensor): filters taps of shape
        :math:`(\text{out\_features}, \text{edge\_features}, \text{neighborhood}+1, \text{in\_features})`
        bias (Tensor): bias of shape
        :math:`(\text{out\_features}, 1)`. Default: :arg:None.
    """
    # The basic idea of what follows is to start reshaping the input and the
    # graph matrix so the filter coefficients go just as a very plain and
    # simple linear operation, so that all the derivatives and stuff on them
    # can be easily computed.

    # input is of shape
    #   batchSize x tSamples x Fin x nNodes
    # graph is of shape
    #   batchSize x tSamples x Fedges x nNodes x nNodes
    # weight is of shape
    #   Fout x Fedges x (Khop+1) x Fin
    # bias (if defined) is of shape
    #   Fout x 1

    # Check that the above dimensions are satisfied
    assert len(input.shape) == 4
    assert len(graph.shape) == 5
    assert len(weight.shape) == 4
    if bias is not None:
        assert len(bias.shape) == 2

    # Let's get those values which we will need to use often
    batchSize = input.shape[0]
    tSamples = input.shape[1]
    Fin = input.shape[2]
    nNodes = input.shape[3]

    # The point here is that the first two dimensions of graph may be equal
    # to 1 and thus different from batchSize and tSamples --it would be
    # pointless to generate such an excess of memory--
    if not graph.shape[0] == batchSize:
        # If we do not have a graph that changes along the batch dimension,
        # it has to be 1
        assert graph.shape[0] == 1
    if not graph.shape[1] == tSamples:
        assert graph.shape[1] == 1
    Fedges = graph.shape[2]
    assert graph.shape[3] == graph.shape[4] == nNodes

    # And the weight
    Fout = weight.shape[0]
    assert weight.shape[1] == Fedges
    Khop = weight.shape[2]-1
    assert weight.shape[3] == Fin

    # And the bias (if exists)
    if bias is not None:
        assert bias.shape[0] == Fout
        assert bias.shape[1] == 1

    # To compute a graph filter, the first thing to do is to compute the
    # sequence graph^k input. This sequence involves a matrix multiplication
    # We can do that by putting all the remaining dimensions to the left and
    # leaving only nNodes x nNodes for graph^k and nNodes x Fin for input

    # Needless to say that we will not be computing powers of graph, we are
    # going to repeatedly multiply graph with the previous graph * input, that
    # is: graph^k input = graph (graph^(k-1) input)
    # Given the sparsity of graph, if a sparse implementation of the matrix
    # multiplication exists, this should speed things up.

    # So, with this idea in mind, note that input has the following "spare"
    # dimensions: batchSize and tSamples
    # While graph has the following "spare" dimensions: batchSize, tSamples and
    # Fedges

    # Thus, we need to transform the input to be
    #   batchSize x tSamples x 1 (for Fedges) x nNodes x Fin
    # First, transpose it so that Fin is the last dmiension
    input = input.permute(0, 1, 3, 2) # batchSize x tSamples x nNodes x Fin
    # Then, add the dimension for Fedges
    input = input.unsqueeze(2) # batchSize x tSamples x 1 x nNodes x Fin
    # While the graph has to be of shape
    #   batchSize x tSamples x Fedges x nNodes x nNodes
    # (which is the same shape, nothing changes)

    # For the output of graph^k input, we are going to accumulate it for each
    # value of k, so we need to create that dimension as well
    # That is, the output of graph^k input will have dimension
    #   batchSize x tSamples x Fedges x 1 (for k) x nNodes x Fin
    # This is basically the dimensions of x (which happens to be the value for
    # k = 0). So we just need to add a further dimension there
    output = input.unsqueeze(3) # batchSize x tSamples x 1 x 1 x nNodes x Fin
    #   Here, the first 1 is for the edge feature dimension and the second  '1'
    #   is for the 'k' dimension
    # And this value is for all edges (is the multiplication with the identity)
    # so we need to repeat this value across all edge features
    output = output.repeat(1, 1, Fedges, 1, 1, 1)
    #   batchSize x tSamples x Fedges x 1 (for k) x nNodes x Fin

    # Now, we're in good shape to start doing the matrix multiplication
    for k in range(Khop):
        # input <- graph * input
        input = torch.matmul(graph.permute(0,1,2,4,3), input)
        #   batchSize x tSamples x Fedges x nNodes x Fin
        #   Note: the permutation comes from the fact that, when calling the
        #   function, the graph and the input are determined row-wise, while
        #   when doing the multiplication, we're doing it column-wise, thus
        #   we not only need to permute the input (which we did before), but
        #   we also need to permute row and columns (i.e. transpose) of the
        #   graph matrix
        # Concatenate it to the output along the k dimension
        output = torch.cat((output,
                            input.unsqueeze(3)),
        #   By doing that unsqueeze it becomes of shape
        #       batchSize x tSamples x Fedges x 1 (for k) x nNodes x Fin
                           dim = 3)
        # After the concatenation, the variable output is of shape
        #   batchSize x tSamples x Fedges x k x nNodes x Fin
    # After the for loop, we end up with an output that has shape
    #   batchSize x tSamples x Fedges x (Khop+1) x nNodes x Fin
    # representing each instance of graph^k input for each k

    # Now it's time to do the multiplication with weights and add over the
    # dimension of Khop, the dimension of Fin and the dimension of Fedges
    # Recall that weight has shape Fout x Fedges x (Khop+1) x Fin
    # This means we want to put all the Fedges, Khop and Fout dimensions
    # together in both output and weight
    # First put the Fedges x (Khop+1) x Fin at the end:
    output = output.permute((0,1,4,2,3,5))
    #   Shape: batchSize x tSamples x nNodes x Fedges x (Khop+1) x Fin
    # And put them all together
    output = output.reshape((batchSize, tSamples, nNodes,
                             Fedges * (Khop+1) * Fin))
    # And do the same for the weights
    weight = weight.reshape((Fout, Fedges * (Khop+1) * Fin))
    #   Shape: Fout x (Fedges * (Khop+1) * Fin)
    # And put the Fout outside so that it can be multiplied as input weight
    weight = weight.permute((1, 0)) # (Fedges * (Khop+1) * Fin) x Fout

    # And finally we can multiply
    output = torch.matmul(output, weight)
    #   Shape: batchSize x tSamples x nNodes x Fout
    # Return them in the expected shape
    output = output.permute(0, 1, 3, 2) # batchSize x tSamples x Fout x nNodes

    # If there is bias, add it
    if bias is not None:
        output = output + bias
    return output

class GraphConv(nn.Module):
    __doc__ = r"""Applies a graph convolution (polynomial/FIR implementation)
    to an input graph signal.

    Let :math:`S` be an :math:`(F_{\text{edges}}, N, N)` graph matrix
    description (GMD), also known as graph shift operator (GSO), where
    :math:`N` is the number of nodes. The value of :math:`F_{\text{edges}}`
    determines the dimension of the edge features
    (e.g., :math:`F_{\text{edges}} = 1` corresponds to scalar weights).
    Denote by :math:`S_{e}` the :math:`(N, N)` matrix corresponding to the
    :math:`e`th edge feature entry.

    Let :math:`X` be a :math:`F_{\text{in}} x N` input graph signal where
    each row collects the value of the :math:`f`th feature across all nodes,
    and where each column collects the value of all :math:`F` features at each
    node.

    The graph convolution is, by definition, given by
    .. math::
        \sum_{e=1}^{F_{\text{edges}}} \sum_{k=0}^{K} H_{e,k} X S_{e}^{k}
    where a bias can be added. This graph convolution maps an input graph
    signal of shape :math:`(F_{\text{in}},N)` with :math:`F_{\text{in}}` input
    features, into another graph signal of shape :math:`(F_{\text{out}},N)`
    with :math:`F_{\text{out}}` output graph features.

    The matrices :math:`H_{e,k}` of shape :math:`(F_{\text{out}},F_{\text{in}})`
    represent the learnable parameters, which can be conveniently collected
    into the :attr:`weight` which has shape
    :math:`(F_{\text{out}}, F_{\text{edges}}, K+1, F_{\text{in}})`."""+ r"""

    * :attr:`set_graph`(:attr:`graph`) sets the graph to use. The variable
    :attr:`graph` is a :attr:`torch.Tensor` of shape
    :math:`(F_{\text{edges}}, N, N)` or :math:`(B, F_{\text{edges}}, N, N)`
    where :math:`B` is the batch size. If the latter is used, it is assumed
    that each signal in the input tensor has a corresponding graph in the
    graph tensor. Note that this allows to change the graph on which the
    convolution is applied, without changing the weights.
    Furthermore, the value of :math:`N` (the size of the graph) may be
    different between graphs.

    Args:
        in_features (int): Number of features in the input graph signal
        out_featres (int): Number of features in the output graph signal
        nb_size (int): Neighborhood size (order of the polynomial),
            i.e. :math:`K`
        edge_features (int): Number of edge features. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``

    Shape:
        - Input: :math:`(B, F_{\text{in}}, N)` where :math:`B` is the batch
        size
        - Output: :math:`(B, F_{\text{out}}, N)`
        - Graph: :math:`(F_{\text{edges}}, N, N)` or
        `(B, F_{\text{edges}}, N, N)` through the method .set_graph(graph)

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
            :math:`(\text{out\_features},\text{edge\_features},
            \text{nb\_size}+1, \text{in\_features})`
        bias (Tensor):   the learnable bias of the module of shape
            (out_channels).

    Note:
        The graph convolution is able to handle time-varying signals as well
        as batch-varying and time-varying graphs.

        More specifically, the input can be of shape
        :math:`(B,T,F_{\text{in}},N)` where :math:`B` is the batch size and
        where :math:`T` is the length of the trajectory under consideration.
        In this case, the output will have shape :math:`(B,T,F_{\text{out}},N)`
        and for each value :math:`t` along the :math:`T` dimension, the graph
        convolution is computed. This assumes that all the neighborhood
        exchanges involved happen for :math:`t` before moving to the next value
        of :math:`t`.

        The graph matrix description also admits several different forms.
        It can be of shape :math:`(F_{\text{edges}}, N, N)` where the same
        graph is considered for all input signals in the batch (and for all
        time-instants if the inputs are trajectories).

        It can be of shape :math:`(B, F_{\text{edges}}, N, N)` where there is
        a different graph matrix for each of the signals in the batch (with a
        one-to-one correspondence).

        It can be of shape :math:`(B, T, F_{\text{edges}}, N, N)` where a
        different graph is applied for each element in the batch, for each
        time instant in the trajectory. This is only valid if the input also
        has a time dimension.

        It can be of shape :math:`(T, F_{\text{edges}}, N, N)` where there is
        a different graph used for each time instant, but where all the grpahs
        in the trajectory are used the same for all signals in the batch.
        Important observation: If :math:`B=T`, then entering a graph matrix
        description of shape :math:`(T, F_{\text{edges}}, N, N)` will be
        automatically interpreted as having shape
        :math:`(B, F_{\text{edges}}, N, N)`. So, if time-varying graph signals
        are going to be used, all supported by the same graph, but changing
        through time, it is recommended that the graph is a tensor of shape
        :math:`(1, T, F_{\text{edges}}, N, N)` to avoid confusion.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 nb_size: int,
                 edge_features: int = 1,
                 bias: bool = True) -> None:

        # Graph matrix description will be added later

        # Initialize parent
        super().__init__()
        # Save parameters:
        self.Fin = in_features
        self.Fout = out_features
        self.Khop = nb_size
        self.Fedges = edge_features
        self.graph = None # No graph assigned yet
        self.nNodes = None # No graph assigned yet
        self.batchSize = None # No graph nor input assigned yet
        self.tSamples = None # No graph nor input assigned yet
        # Create parameters:
        self.weight = nn.parameter.Parameter(torch.Tensor(self.Fout,
                                                          self.Fedges,
                                                          self.Khop+1,
                                                          self.Fin))
        if bias:
            self.bias = nn.parameter.Parameter(torch.Tensor(self.Fout, 1))
        else:
            self.register_parameter('bias', None)
        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Taken from _ConvNd initialization of parameters:
        nn.init.kaiming_uniform_(self.weight, a = math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)


    def set_graph(self, graph: torch.Tensor) -> None:
        # graph is the graph matrix description (GMD), also known as graph
        # shift operator (GSO). It is essentially any matrix that respects
        # the sparsity of the graph (most commonly used are normalized version
        # of the adjacency or Laplacian matrix)

        # The graph matrix can have anywhere from 3 to 5 dimensions.

        if len(graph.shape) == 5:
            # If it has dimension 5 is is of the form B x T x E x N x N
            # B: batch size
            # T: trajectory duration
            # E: edge features
            # N: number of nodes
            self.batchSize = graph.shape[0]
            self.tSamples = graph.shape[1]
            assert graph.shape[2] == self.Fedges
            assert graph.shape[3] == graph.shape[4]
            self.nNodes = graph.shape[3]
        elif len(graph.shape) == 4:
            # Here, we can have two options, the first dimension is either the
            # batch dimension or the time dimension
            self.batchSize = graph.shape[0] # It could be either this one
            self.tSamples = graph.shape[0] # or this one
            assert graph.shape[1] == self.Fedges
            assert graph.shape[2] == graph.shape[3]
            self.nNodes = graph.shape[2]
        elif len(graph.shape) == 3:
            # If it has size 3, then it is easily E x N x N
            assert graph.shape[0] == self.Fedges
            assert graph.shape[1] == graph.shape[2]
            self.nNodes = graph.shape[1]

        self.graph = graph

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # First check we have a graph, otherwise, we cannot do anything
        assert self.graph is not None
        # x can be of shape B x T x Fin x N or B x Fin x N
        if len(input.shape) == 4:
            # In this context, we now that the input has a time dimension
            # Check the dimensions
            assert input.shape[2] == self.Fin
            assert input.shape[3] <= self.nNodes
            # A flag that indicates the input originally has a time dimension
            hasTimeDimension = True
            # Check the dimension of the graph
            if len(self.graph.shape) == 3:
                # In this case, we have the same graph E x N x N, for all
                # B and for all T, so we just need to add the dimensions
                graph = self.graph.reshape((1, 1, self.Fedges,
                                            self.nNodes, self.nNnodes))
            elif len(self.graph.shape) == 4:
                # In this case, we have a fourth dimension that can either
                # be a time dimension B x E x N x N or a time dimension
                # T x E x N x N.
                # We have no way of checking which one is which, so we will
                # see which dimensions match.
                if self.graph.shape[0] == self.input.shape[0]:
                    # In this case, the first dimension of the graph, matches
                    # the batchsize, so that's what we assume is happening
                    # We adapt the graph to add the time dimension
                    graph = self.graph.reshape((self.graph.shape[0], 1,
                                                self.Fedges,
                                                self.nNodes, self.nNodes))
                elif self.graph.shape[0] == self.input.shape[1]:
                    # In this case, the first dimension of the graph, matches
                    # the tSamples size, so we have one graph for all samples
                    graph = self.graph.reshape((1, self.graph.shape[0],
                                                self.Fedges,
                                                self.nNodes, self.nNodes))
                else:
                    # If we are here, the dimensions failed
                    print("WARNING: Dimensions do not check out.")
            elif len(self.graph.shape) == 5:
                # In this case, the graph has all dimensions B x T x E x N x N
                # so we just need to see if they check out
                assert input.shape[0] == self.graph.shape[0] # batchSize
                assert input.shape[1] == self.graph.shape[1] # tSamples
                # And save the graph
                graph = self.graph

        elif len(input.shape) == 3:
            # Now, let's check the case when the input dimension is just
            # B x F x N
            # Check the dimensions
            assert input.shape[1] == self.Fin
            assert input.shape[2] <= self.nNodes
            # A flag to indicate there was no time dimension in the input
            hasTimeDimension = False
            # And check the graph dimensions
            if len(self.graph.shape) == 3:
                # In this case, we just have a graph of shape E x N x N, so
                # nothing big nothing strange to do
                graph = self.graph.reshape((1, 1, self.Fedges,
                                            self.nNodes, self.nNodes))
            elif len(self.graph.shape) == 4:
                # In this case, necessarily, we need to have it be the batch
                # dimension, as there is no time in this problem B x E x N x N
                assert self.graph.shape[0] == input.shape[0]
                # And we still need to add the time dimension, as the function
                # that we use to compute the output filter is the same
                # for all cases
                graph = self.graph.reshape((self.graph.shape[0], 1,
                                            self.Fedges,
                                            self.nNodes, self.nNodes))
            elif len(self.graph.shape) == 5:
                # In this case, the graph is just E x N x N, so we just need
                # to add the two remaining dimensions
                graph = self.graph.reshape((1, 1, self.Fedges,
                                            self.nNodes, self.nNodes))
            # And now we also need to add the remaining dimension for the time
            # so that it can be adequately processed by the functional
            input = input.reshape((input.shape[0], 1,
                                   input.shape[1], input.shape[2]))


        # After all of these, we have two variables
        #   input: batchSize x tSamples x Fin x nIn
        #   graph: batchSize x tSamples x Fedges x nNodes x nNodes

        batchSize = input.shape[0]
        tSamples = input.shape[1]
        nIn = input.shape[3]

        # If the size of the input is less than t he GSO, then we add zeros
        # for zero-padding (it may be due to clustering)
        if nIn < self.nNodes:
            input = torch.cat((input,
                               torch.zeros(batchSize, tSamples,
                                           self.Fin, self.nNodes-nIn,
                                           dtype = input.dtype,
                                           device = input.device)),
                              dim = 3)
        # Compute the filter output
        output = graphConv(input, graph, self.weight, self.bias)
        # So far, the output has shape batchSize x tSamples x Fout x nNodes
        # And we want to return a tensor of shape
        # batchSize x tSamples x Fout x nIn
        # since the nodes between nIn and nNodes are not required (they were
        # put there from zero-oadding)
        if nIn < self.nNodes:
            output = output[:, :, :, 0:nIn]
        # Get removed of the time dimension if the input didn't have it in the
        # first place
        if not hasTimeDimension:
            output = output.squeeze(1) # Now it will be B x Fout x nNodes
        return output

    def extra_repr(self):
        s =  'in_features = %d, ' % self.Fin
        s += 'out_features = %d, ' % self.Fout
        s += 'nb_size = %d, ' % self.Khop
        s += 'edge_features = %d, ' % self.Fedges

        if self.graph is not None:
            s += "graph stored (%d nodes)" % self.nNodes
        else:
            s += "no graph stored"
        return s