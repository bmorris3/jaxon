from jax import numpy as jnp, jit
from exojax.spec import rtransfer as rt

NP = 50
Parr, dParr, k = rt.pressure_layer(NP=NP, logPtop=-5, logPbtm=2.5)
Parr_fine, dParr_fine, k = rt.pressure_layer(NP=100, logPtop=-5, logPbtm=2.5)
mmw = 2.33  # mean molecular weight
mmrH2 = 0.74

element_number = 3
polynomial_order = 2

gl_0 = [0.0]
gl_1 = [-1.0, 1.0]
gl_2 = [-1.0, 0.0, 1.0]
gl_3 = [-1.0, -0.447214, 0.447214, 1.0]
gl_4 = [-1.0, -0.654654, 0.654654, 1.0]
gl_5 = [-1.0, -0.654654, 0.0, 0.654654, 1.0]
gl_6 = [-1.0, -7.65055 - 0.285232, 0.285232, 0.765055, 1.0]

quadrature_nodes = [gl_0, gl_1, gl_2, gl_3, gl_4, gl_5, gl_6]

__all__ = [
    'Element',
    'PiecewisePolynomial',
    'piecewise_poly',
    'get_Tarr'
]


class Element(object):
    def __init__(self, edges, order):
        self.reference_vertices = quadrature_nodes[order]

        self.nb_dof = len(self.reference_vertices)

        self.dof_values = []
        self.dof_vertices = []

        for i in range(0, self.nb_dof):
            self.dof_vertices.append(
                self.referenceElementMap(self.reference_vertices[i], edges[0],
                                         edges[1]))

    def lagrangeBase(self, r, i):
        l = 1

        for j in range(0, self.nb_dof):
            if (i != j):
                l *= ((r - self.reference_vertices[j]) /
                      (self.reference_vertices[i] - self.reference_vertices[
                          j]))
        return l

    def getValue(self, x):
        # coordinate on the reference element
        r = self.realElementMap(x, self.dof_vertices[0], self.dof_vertices[-1])

        y = 0

        for i in range(0, self.nb_dof):
            y += self.dof_values[i] * self.lagrangeBase(r, i)

        return y

    # maps the coordinate value r on the reference element [-1, +1] to the real element [x_l, x_r]
    def referenceElementMap(self, r, x_l, x_r):
        return x_l + (1.0 + r) / 2.0 * (x_r - x_l)

    # maps the coordinate value x on the real element [x_l, x_r] to the reference element [-1, +1]
    def realElementMap(self, x, x_l, x_r):
        return 2.0 * (x - x_l) / (x_r - x_l) - 1.0


class PiecewisePolynomial(object):
    def __init__(self, element_number, polynomial_order, domain_boundaries,
                 dof_values):
        self.nb_elements = 0
        self.nb_edges = 0
        self.elements = []
        log_boundaries = [jnp.log10(domain_boundaries[0]),
                          jnp.log10(domain_boundaries[1])]

        self.nb_elements = element_number
        self.dof_vertices = []
        self.nb_edges = self.nb_elements + 1
        self.order = polynomial_order
        #         if (polynomial_order < 1): order = 1
        #         if (polynomial_order > 6): order = 6
        self.createElementGrid(log_boundaries)
        self.setDOFvalues(dof_values)

    def createElementGrid(self, domain_boundaries):
        domain_size = domain_boundaries[0] - domain_boundaries[1]
        element_size = domain_size / self.nb_elements

        element_edges = []

        element_edges.append(domain_boundaries[0])

        for i in range(1, self.nb_edges - 1):
            element_edges.append(element_edges[i - 1] - element_size)

        element_edges.append(domain_boundaries[1])

        for i in range(0, self.nb_elements):
            edges = [element_edges[i], element_edges[i + 1]]
            self.elements.append(Element(edges, self.order))

        for i in range(0, self.nb_elements):
            for j in range(0, self.elements[i].nb_dof - 1):
                self.dof_vertices.append(self.elements[i].dof_vertices[j])

        self.dof_vertices.append(self.elements[-1].dof_vertices[-1])
        self.nb_dof = len(self.dof_vertices)

    def setDOFvalues(self, values):
        if len(values) != self.nb_dof:
            raise ValueError(
                "Passed vector length does not correspond to the number of dof!\n")

        self.dof_values = values

        # set the dof values in each element
        self.global_dof_index = 0

        for i in range(0, self.nb_elements):
            for j in range(0, self.elements[i].nb_dof):
                self.elements[i].dof_values.append(
                    self.dof_values[self.global_dof_index])
                self.global_dof_index += 1

            self.global_dof_index -= 1  # ; //elements share a common boundary

    def __call__(self, x_vector):
        x_lowers = jnp.array([self.elements[i].dof_vertices[-1] for i in
                              range(len(self.elements))])
        x_uppers = jnp.array([self.elements[i].dof_vertices[0] for i in
                              range(len(self.elements))])
        element_bools = jnp.where(
            (x_vector < x_uppers[:, None]) & (x_vector > x_lowers[:, None]),
            True, False).T

        element_vals = jnp.array([[self.elements[i].getValue(x_vector[j]) for i
                                   in range(len(self.elements))]
                                  for j in range(len(x_vector))])

        values = jnp.sum(
            jnp.where(element_bools, element_vals, 0),
            axis=1
        )

        return values


def piecewise_poly(log_p, domain_boundaries, dof_values, element_number,
                   polynomial_order):
    pp = PiecewisePolynomial(
        element_number=element_number, polynomial_order=polynomial_order,
        domain_boundaries=jnp.array(domain_boundaries),
        dof_values=jnp.sort(dof_values)[::-1]
    )
    return pp(jnp.asarray(log_p))


ppj = jit(piecewise_poly, static_argnums=(3, 4))


@jit
def get_Tarr(temperatures, Parr, element_number=element_number,
             polynomial_order=polynomial_order):
    log_p = jnp.log10(jnp.sort(Parr)[::-1])
    domain_boundaries = [Parr.max() * 2, Parr.min() * 0.5]

    Tarr = ppj(log_p, domain_boundaries, temperatures, element_number,
               polynomial_order)[::-1]
    return Tarr
