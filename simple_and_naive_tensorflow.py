# coding=utf-8
# 元类没有使用，只能计算标量
from pprint import pprint
_varibles = set()
_graph = set()

class Node:
    def __init__(self, *args):
        self.args = args

        self.children = set()
        self.parents = set()
        
        _graph.add(self)
    
    def eval(self, feed_dict=None):
        raise NotImplementedError()

    @property
    def op(self):
        raise NotImplementedError()

    def __mul__(self, other):
        return mul(self, other)

    def __sub__(self, other):
        return sub(self, other)

    def __repr__(self):
        return "%s(%s)"%(self.name if hasattr(self, 'name') else self.__class__.__name__, self.value)

class Variable(Node):
    """Variable"""
    def __init__(self, value: int, name:str, *args):
        super().__init__(*args)
        self.value = value
        self.name = name

        _varibles.add(self)

    def eval(self, feed_dict=None):
        return self.value

class Constant(Node):
    def __init__(self, value, *args):
        super().__init__(*args)
        self.value = value
        self.args = args

    def eval(self, feed_dict=None):
        return self.value
    
    def __str__(self):
        return "%s"%self.value
    
class Operation(Node):
    def __init__(self, name, *args):
        super().__init__(*args)
        self.name = name
        self.args = args

        for p in args:
            p.children.add(self)
            self.parents.add(p)
        
    @property
    def op(self):
        return self# .__class__
 
    def bprop(self, I, V, D):
        raise NotImplementedError(repr(self))
    
    def eval(self, feed_dict=None):
        raise NotImplementedError()
    def __str__(self):
        try:
            return "%s(%s)"%(self.__class__.__name__, str(self.args))
        except:
            return "%s()"%(self.__class__.__name__)
    
class AddOperation(Operation):
    def eval(self, feed_dict=None):
        return self.args[0].eval(feed_dict)+self.args[1].eval(feed_dict)

    def bprop(self, I, V, D):
        """ p178 (6.47)    p186 (6.54)"""
        # return Constant(1) * D
        return D
        
    def __repr__(self):
        return "%s + %s"%self.args
class SubOperation(Operation):
    def eval(self, feed_dict=None):
        return self.args[0].eval(feed_dict)-self.args[1].eval(feed_dict)
    def bprop(self, I, V, D):
        index = [i for i, k in enumerate(I) if k is V]
        if index == [1]:
            return D * Constant(-1)
        elif index == [0]:
            return D
        else:
            pprint(self)
            raise ValueError(index)

    def __str__(self):
        return "%s - %s"%self.args
    
class MulOperation(Operation):
    def eval(self, feed_dict=None):
        return self.args[0].eval(feed_dict)*self.args[1].eval(feed_dict)
    
    def bprop(self, I, V, D):
        """ p178 (6.47)    p186 (6.54)"""
        index = [i for i, k in enumerate(I) if k is V]
        if index == [1]:
            return self.args[0]*D
        elif index == [0]:
            return self.args[1]*D
        elif index == [0, 1]:
            return self*Constant(2)*D
        else:
            pprint(self)
            raise ValueError(index)
    def __str__(self):
        return "%s * %s"%self.args
        
class DivOperation(Operation):
    def eval(self, feed_dict=None):
        return self.args[0].eval(feed_dict)/self.args[1].eval(feed_dict)
    def __str__(self):
        return "%s / %s"%self.args
class PowOperation(Operation):
    def eval(self, feed_dict=None):
        return self.args[0].eval(feed_dict)**self.args[1].eval(feed_dict)
    def bprop(self, I, V, D):
        """ p178 (6.47)    p186 (6.54)"""
        index = [i for i, k in enumerate(I) if k is V]
        if index == [1]:
            return self.args[0]*D
        elif index == [0]:
            return self.args[1]*pow_(self.args[0], self.args[1]-Constant(1))*D
        elif index == [0, 1]:
            raise NotImplementedError(self)
        else:
            pprint(self)
            raise ValueError(index)
    def __str__(self):
        return "%s ** %s"%self.args
    
class SqrtOperation(Operation):
    def eval(self, feed_dict=None):
        return self.args[0].eval(feed_dict)**2
    
    def bprop(self, I, V, D):
        return Constant(-0.5) * pow_(self, Constant(-1)) * D
    def __str__(self):
        return "√(%s)"%self.args
class AssignOperation(Operation):
    def __init__(self, name, *args):
        # super().__init__(*args)
        self.name = name
        self.args = args
        
    def eval(self, feed_dict=None):
        # print(self.args)
        for V, value in zip(*self.args):
            V.value = value.eval(feed_dict) if hasattr(value, 'eval') else value

    def __str__(self):
        res=[]
        for V, value in zip(*self.args):
            res.append("%s := %s"%(V, value))
        return "\n".join(res)
        
        
class PlaceHolder(Node):
    def __init__(self, tp, name, *args):
        super().__init__(*args)
        self.tp = tp
        self.name = name
        self.args = args
        
    def eval(self, feed_dict=None):
        return feed_dict[self]
    
    def __repr__(self):
        return "%s(%s)"%(self.__class__.__name__, self.name)

    def __str__(self):
        return "%s"%self.name
placeholder = PlaceHolder

def add(op1, op2):
    return AddOperation('add', op1, op2)

def sub(op1, op2):
    return SubOperation('sub', op1, op2)

def mul(op1, op2):
    return MulOperation('mul', op1, op2)

def pow_(op1, op2):
    return PowOperation('mul', op1, op2)

def square(op):
    return PowOperation('square', op, Constant(2))

def assign(V, VV):
    return AssignOperation('assign', V, VV)

class GradientDescentOptimizer:
    def __init__(self, lr):
        self.lr = Constant(lr)
    def minimize(self, z):
        varibles = list(_varibles)
        g = make_grad_table(varibles, _graph, z)
        _op_value = [k - self.lr * g[k] for k in varibles]
        return AssignOperation("assign", varibles, _op_value)

class Session:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __init__(self):
        self.g={}

    def run(self, T, feed_dict=None):
        return [V.eval(feed_dict) for V in T]

def make_grad_table(T, graph, z):
    # print("make_grad_table(T, graph, z)",T, graph, z)
    graph_t = cut_graph(T, graph, z)
    # pprint(graph_t)
    grad_table = {}
    grad_table[z] = Constant(1)
    for V in T:
        build_grad(V, graph, graph_t, grad_table)
    return {k: v for k, v in grad_table.items() if k in T}

def build_grad(V, graph, graph_t, grad_table):
    if V in grad_table:
        return V

    ggg = []
    for C in get_consumers(V, graph_t):
        grad_C = build_grad(C, graph, graph_t, grad_table)
        inputs = get_inputs(C, graph_t)
        v = C.op.bprop(inputs, V, grad_C)
        ggg.append(v)


    it = iter(ggg)
    grad = next(it)
    for else_ in it:
        grad = add(grad, else_)
        
##    grad = sum(C.op.bprop(get_inputs(C, graph_t), 
##                          V, 
##                          build_grad(C, graph, graph_t, grad_table)) 
##               for C in get_consumers(V, graph_t))

    
    grad_table[V] = grad
    # graph.add(grad)
    return grad

def cut_graph(T, graph, z):
    if graph is None:
        graph = _graph
    ans = get_ancestors(z)
    des = {d  for v in T for d in get_decestors(v)}
        
    graph_t = {k for k in graph if k in ans or k in des}
    #graph_t.add(z)
    #graph_t.update(T)
    
    return graph_t

def get_ancestors(V):
    res = set()
    for a in V.parents:
        res.update(get_ancestors(a))
    res.update(V.parents)
    return res

def get_decestors(V):
    res = set()
    for a in V.children:
        res.update(get_decestors(a))
    res.update(V.children)
    return res

def get_consumers(V, graph_t):
    return [k for k in V.children if k in graph_t]

def get_inputs(V, graph_t):
    return [k for k in V.args if k in graph_t]

def initialize_all_variables():
    return []

if __name__=='__main__':
    X = placeholder("float", 'X')
    w = Variable(77.0, name="weight")
    z=X*w
    print(z.eval({X:2}))
    g=GradientDescentOptimizer(0.01)
    zz=g.minimize(z)
