import gym
import numpy as np

from gym import spaces
from tabulate import tabulate


class VacunaEnv(gym.Env):
    """ VacunaEnv
    Clase que define el comportamiento del sistema para la vacunacion por COVID.
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self, ns: int, e_min: int, e_max: int, s_nulas: int, d_max: int) -> None:
        """Constructor del entorno
        Args:
            ns (int): numero de semanas máximas a vacuanar
            e_min (int): cantidad de vacunas mas pequeña a suministrar
            e_max (int): cantidad de vacunas mas grande a suministrar
            s_nulas (int): numero de semanas sin vacunas
            d_max (int): dosis máximas a suministrar
        """

        super(VacunaEnv, self).__init__()
        self.TIEMPO_2_DOSIS = 3 
        self.ns, self.e_min, self.e_max, self.s_nulas, self.d_max = ns, e_min, e_max, s_nulas, d_max
        assert self.ns > 3, "El numero de semanas debe ser mayor que 3"
        assert self.s_nulas < self.ns - self.TIEMPO_2_DOSIS, "El numero de semanas con entrega nula no puede ser mayor que s_max"
        assert self.d_max > self.ns - self.s_nulas, "Al menos 1 dosis por cada semana no nula"
        # Variables de control del entorno
        self.done = False
        self.done = False
        self.sa = 0
        # Creacion de las variables
        (self.p_a_vacunar,
         self.stock,
         self.p_vac_1_d,
         self.p_tot_vac_1_d,
         self.p_vac_2_d,
         self.p_tot_vac_2_d,
         self.p_a_vac_2_d) = self.crear_entregas()

        # compatibilidad con paquete gym
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high= self.stock.sum() / 2,
                                            shape=(9,), dtype=np.float32)

    def crear_entregas(self) -> tuple:
        """Funcion que crea una entrega aleatoria basandose en los parametros
           pasados al constructor
        Returns:
            tuple: 
                - p_a_vacunar: personas totales a vacunar
                - stock: stock total de vacunas por semana
                - p_vac_1_d: personas a vacunar en primera dosis por semana [0,...,0]
                - p_tot_vac_1_d: personas total vacunadas en primera dosis por semana [0,...,0]
                - p_vac_2_d: personas a vacunar en segunda dosis por semana [0,...,0]
                - p_tot_vac_2_d: personas total vacunadas en segunda dosis por semana [0,...,0]
                - p_a_vac_2_d: personas a vacunar en segunda dosis por semana [0,...,0]
        """
        # Creacion de vector de entregas
        entregas = np.ones(self.ns, dtype=int)
        entregas[-self.TIEMPO_2_DOSIS:] = 0
        entregas[np.random.choice((self.ns - self.TIEMPO_2_DOSIS), self.s_nulas, replace=False)] = 0
        entregas = [x if x == 0 else int(np.random.choice(range(self.e_min, self.e_max + 1, 2), 1)) for x in entregas]
        # escalar para obtener el numero de entregas correctas
        entregas = np.array(np.array(self.d_max) * entregas / np.sum(entregas), dtype=int)
        if np.sum(entregas) != self.d_max:
            entregas[-4] += self.d_max - np.sum(entregas)
        # Variables de estado del entorno
        p_a_vacunar = np.sum(entregas) / 2
        stock = np.array(entregas, dtype=np.float)
        p_vac_1_d = np.zeros(self.ns)
        p_tot_vac_1_d = np.cumsum(p_vac_1_d)
        p_vac_2_d = np.zeros(self.ns)
        p_tot_vac_2_d = np.cumsum(p_vac_2_d)
        p_a_vac_2_d = np.zeros(self.ns)
        return p_a_vacunar, stock, p_vac_1_d, p_tot_vac_1_d, p_vac_2_d, p_tot_vac_2_d, p_a_vac_2_d

    def reset(self) -> tuple:
        """Resetea el entorno
        Returns:
            observation (tuple): estado del entorno formado por las siguientes variables
                - personas a vacunar
                - personas a vacunar
                - personas total vacunadas en primera dosis
                - personas total vacunadas en segunda dosis
                - stock proxima semana
                - stock proxima proxima semana
                - stock proxima proxima proxima semana
                - personas a vacunar en 2 dosis en la proxima semana
                - personas a vacunar en 2 dosis en la proxima proxima semana
                - personas a vacunar en 2 dosis en la proxima proxima proxima semana
        """
        self.done = False
        self.sa = 0
        (self.p_a_vacunar,
         self.stock,
         self.p_vac_1_d,
         self.p_tot_vac_1_d,
         self.p_vac_2_d,
         self.p_tot_vac_2_d,
         self.p_a_vac_2_d) = self.crear_entregas()

        observation = (self.p_a_vacunar, 
                        self.p_tot_vac_1_d[self.sa],
                        self.p_tot_vac_2_d[self.sa],

                        self.stock[self.sa + 1] if self.sa < (self.ns - 1) else 0,
                        self.stock[self.sa + 2] if self.sa < (self.ns - 2) else 0,
                        self.stock[self.sa + 3] if self.sa < (self.ns - 3) else 0,

                        self.p_a_vac_2_d[self.sa + 1] if self.sa < (self.ns - 1) else 0,
                        self.p_a_vac_2_d[self.sa + 2] if self.sa < (self.ns - 2) else 0,
                        self.p_a_vac_2_d[self.sa + 3] if self.sa < (self.ns - 3) else 0)
        return observation

    def vacunar(self, action: tuple) -> None:
        """Funcion que realiza las acciones indicadas por el agente.
        REGLA 1:
        Cada vez que se vacuna alguien de la 1 dosis debe ser vacunado tres
        semanas mas tarde, salvo que estemos en la ultima semana, que entonces
        no se hace nada. El paciente se pierde
        REGLA 2:
        personas que tenian que ser vacunadas de 2 dosis y que no han
        sido, deben ser pasadas a la siguiente semana, si no es la ultima
        REGLA 3:
        Si quiero vacunar mas que los que tengo tampoco puedo, y me sobraran
        vacunas que pasaran a stock
        REGLA 4:
        Si queda stockaje de vacunas las paso a la semana siguiente

        Args:
            acciones (tuple): acciones: tupla de acciones con el formato
            (porc_vac_1d, porc_vac_2d, porc_stock)
            - porc_vac_1d: Porcentaje del stock de la semana destinada a vacunas 1 dosis
            - porc_vac_2d: Porcentaje del stock de la semana destinada a vacunas 2 dosis
            - porc_stock: Porcentaje que va a stockaje
        """

        porc_vac_1d, porc_vac_2d, porc_stock = action
       # REGLA 1
        # Vacunacion primera dosis
        self.p_vac_1_d[self.sa] += porc_vac_1d * self.stock[self.sa]
        if self.sa >= (self.ns - 3):
            pass
        else:
            self.p_a_vac_2_d[self.sa + 3] += porc_vac_1d * self.stock[self.sa]
        # Vacunacion segunda dosis
        self.p_vac_2_d[self.sa] += porc_vac_2d * self.stock[self.sa]
        
        # Pasar al almacenamiento
        if self.sa != self.ns - 1:
            self.stock[self.sa + 1] += porc_stock * self.stock[self.sa]
        #else:
        #    self.stock[self.sa] += porc_stock * self.stock[self.sa]
        # Actualizar el stock
        stock = self.stock[self.sa]
        self.stock[self.sa] -= porc_vac_2d * stock
        self.stock[self.sa] -= porc_vac_1d * stock
        self.stock[self.sa] -= porc_stock * stock #  stock[sa] = 0
        # REGLA 4
        if self.sa != self.ns - 1:
            self.stock[self.sa + 1] += self.stock[self.sa]
            self.stock[self.sa] = 0

        # Actualizar Acumulados
        self.p_tot_vac_1_d = np.cumsum(self.p_vac_1_d)
        self.p_tot_vac_2_d = np.cumsum(self.p_vac_2_d)

        # REGLA 2
        if self.p_a_vac_2_d[self.sa] > self.p_vac_2_d[self.sa] and self.sa < self.ns - 1:
            self.p_a_vac_2_d[self.sa + 1] += (self.p_a_vac_2_d[self.sa] - self.p_vac_2_d[self.sa])
        # REGLA 3
        if self.p_a_vac_2_d[self.sa] < self.p_vac_2_d[self.sa]:
            if self.sa < self.ns - 1:
                self.stock[self.sa + 1] += (self.p_vac_2_d[self.sa] - self.p_a_vac_2_d[self.sa])
            self.p_vac_2_d[self.sa] = self.p_a_vac_2_d[self.sa]

    def recompensar(self) -> tuple:
        """Funcion que calcula las recompensas
        REGLA 1
        Penalizar si se vacuna 2 dosis en las tres primeras semanas (no se puede)
        REGLA 2
        Penalizar si se vacuna primera dosis en las ultimas 3 semanas
        REGLA 3
        Recompensar por cada persona vacunada en 2 dosis.
        REGLA 4
        Recompensar por cada persona vacunada en 1 dosis.
        REGLA 5
        Penalizar por dejar personas en segunda dosis sin vacunar
        es np.abs porque si quiero vacunar mas que las que debo tambien se
        penaliza
        REGLA 6
        Penalizar si queda stock en la ultima semana

        Returns:
            float: recompensa
        """
        recompensa = 0.
        RECOM_2DOSIS = 1.0
        # Regla 3
        recompensa = self.p_vac_2_d[self.sa] * RECOM_2DOSIS
        return recompensa

    def step(self, action: tuple) -> tuple:
        """Evoluciona la semana segun las instrucciones del agente
        y devuelve el estado del sistema
       
        Args:
            acciones (tuple): (porc_vac_1d, porc_vac_2d, porc_stock)

        Returns:
            observation (tuple): estado del entorno formado por las siguientes variables
                - personas a vacunar
                - personas total vacunadas en primera dosis
                - personas total vacunadas en segunda dosis
                - stock proxima semana
                - stock proxima proxima semana
                - stock proxima proxima proxima semana
                - personas a vacunar en 2 dosis en la proxima semana
                - personas a vacunar en 2 dosis en la proxima proxima semana
                - personas a vacunar en 2 dosis en la proxima proxima proxima semana

            reward (float): recompensa calculada
            done (bool): Terminado
            info: None
        """
        self.vacunar(action=action)
        reward = self.recompensar()

        observation = (self.p_a_vacunar,
                        self.p_tot_vac_1_d[self.sa],
                        self.p_tot_vac_2_d[self.sa],

                        self.stock[self.sa + 1] if self.sa < (self.ns - 1) else 0,
                        self.stock[self.sa + 2] if self.sa < (self.ns - 2) else 0,
                        self.stock[self.sa + 3] if self.sa < (self.ns - 3) else 0,

                        self.p_a_vac_2_d[self.sa + 1] if self.sa < (self.ns - 1) else 0,
                        self.p_a_vac_2_d[self.sa + 2] if self.sa < (self.ns - 2) else 0,
                        self.p_a_vac_2_d[self.sa + 3] if self.sa < (self.ns - 3) else 0)
        # Cambiar de semana
        if self.sa == self.ns - 1:
            self.done = True
        else:
            self.sa += 1
            self.done = False

        info = None
        return observation, reward, self.done, info

    def mostar_estado(self) -> str:
        """Funcion para representacion

        Returns:
            string: Devuelve el estado interno
        """
        table = [["Numero de semanas: ", self.ns],
                 ["Semana actual: ", self.sa],
                 ["Stock Actual: ", self.stock],
                 ["p_vac_1_d: " , self.p_vac_1_d],
                 ["p_vac_2_d: ", self.p_vac_2_d],
                 ["p_tot_vac_1_d: ", self.p_tot_vac_1_d],
                 ["p_tot_vac_2_d: ", self.p_tot_vac_2_d],
                 ["p_a_vac_2_d: ", self.p_a_vac_2_d],
                 ["done: ", self.done]
                ]
        return tabulate(table)

    def render(self, mode="human"):
        """No hay necesidad de implementarlo"""
        pass

    def close(self):
        _ = self.reset()

# Para testear
if __name__ == "__main__":
    srs = VacunaEnv(e_max=50, e_min=12, s_nulas=0, ns=20, d_max=200)
    done = False
    print(srs.mostar_estado())
    while done == False:
        _, reward, done, info = srs.step((0.3, 0.5, 0.2))
        print(reward)
        #print(srs.mostar_estado())
        #input("C continuar?")
