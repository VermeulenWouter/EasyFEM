from typing import Callable

from abaqusConstants import BILAMINA, ENGINEERING_CONSTANTS, LAMINA, ENERGY
from numpy.typing import NDArray
import numpy as np

from utils.abaqus import Model, insert_in_input_file
from utils.units import Quantity, ureg
from utils.prettyprint import tprint


class Material:
    def __init__(self, model: Model, name: str, description: str | None = None):
        self.model = model
        self.material = self.model.Material(name=name, description=description)

        self.alpha = None  # Thermal expansion coefficient

        self.info = {"name": name, "description": description}

    def Density(self, density: Quantity):
        """
        Define density for the material.

        :param density: density of the material (units of ML^-3)

        :source
        [1] Abaqus 2024 documentation, https://docs.software.vt.edu/abaqusv2024/English/SIMACAEMATRefMap/simamat-c-linearelastic.htm#simamat-c-linearelastic-planestressbilamina
        """
        self.material.Density(table=((density.m, ), ))
        self.info["density"] = density.dict

    def Elasticity(self, E: Quantity, nu: Quantity):
        """
        Define elasticity for the material.

        :param E: Young's modulus
        :param nu: Poisson's ratio

        :source
        [1] Abaqus 2024 documentation
        """
        self.material.Elastic(table=((E.m, nu.m), ))
        self.info["elasticity"] = {"E": E.dict, "nu": nu.dict}

    def ExpansionCoefficient(self, alpha: Quantity):
        self.material.Expansion(table=((alpha.m, ), ))
        self.alpha = alpha
        self.info["expansion_coefficient"] = alpha.dict

    @property
    def name(self):
        """The name given to the material at time of creation."""
        return self.material.name


class SMAMaterial(Material):
    def __init__(self, model: Model, name: str, description: str | None = None):
        super().__init__(model, name, description)

        self._superelastic = False

    def SuperElasticity(self, E_M: Quantity, nu_M: Quantity, epsilon_L: Quantity, sigma_tL_S: Quantity,
            sigma_tL_E: Quantity, sigma_tU_S: Quantity, sigma_tU_E: Quantity, sigma_cL_S: Quantity):
        """
        Define superelastic properties for the material.

        *Note: this implementation does not include shape memory effect, or any temperature-related properties, though
        Abaqus does support them. See source below.*

        :param E_M: martensite Young's modulus
        :param nu_M: martensite Poisson's ratio
        :param epsilon_L: transformation strain
        :param sigma_tL_S: stress at the start of transformation during loading
        :param sigma_tL_E: stress at the end of transformation during loading
        :param sigma_tU_S: stress at the start of transformation during unloading
        :param sigma_tU_E: stress at the end of transformation during unloading
        :param sigma_cL_S: stress at the start of transformation during compression

        :source
        [1] Abaqus 2024 documentation, simakey-r-superelastic.htm
        """
        self.material.SuperElasticity(table=((E_M.m, nu_M.m, epsilon_L.m, sigma_tL_S.m, sigma_tL_E.m, sigma_tU_S.m,
                                              sigma_tU_E.m, sigma_cL_S.m, 0),))  # For now reference temp is just 0
        self._superelastic = True
        self.info["superelasticity"] = {"E_M": E_M.dict, "nu_M": nu_M.dict, "epsilon_L": epsilon_L.dict,
            "sigma_tL_S": sigma_tL_S.dict, "sigma_tL_E": sigma_tL_E.dict, "sigma_tU_S": sigma_tU_S.dict,
            "sigma_tU_E": sigma_tU_E.dict, "sigma_cL_S": sigma_cL_S.dict}

    def SuperElasticityHardening(self, yield_stress: list[Quantity], yield_strain: list[Quantity]):
        """
        Define superelastic hardening behaviour for the material.

        :param yield_stress: yield stress for the superelastic hardening behaviour
        :param yield_strain: yield strain for the superelastic hardening behaviour

        :source
        [1] Abaqus 2024 documentation, simakey-r-superelastichardening.htm
        """
        if not self._superelastic:
            raise AssertionError("Material does not have superelastic behaviour defined. Please define superelastic"
                                 " behaviour first before defining superelastic hardening behaviour.")
        self.material.superElasticity.SuperElasticHardening(
            table=tuple((stress.m, strain.m) for stress, strain in zip(yield_stress, yield_strain)))
        self.info["superelastic_hardening"] = {"yield_stress": [stress.dict for stress in yield_stress],
            "yield_strain": [strain.dict for strain in yield_strain]}


class FRPMaterial(Material):
    def __init__(self, model: Model, name: str, description: str | None = None):
        super().__init__(model, name, description)
        self._elasticity = None
        self._damage_initiation_criterion = None
        self._damage_evolution = None
        self._ply_fabric_initiation_number_of_lines = None

    def IsotropicElasticity(self, E: Quantity, nu: Quantity):
        """
        Define elasticity for the material.

        :param E: Young's modulus
        :param nu: Poisson's ratio

        :source
        [1] Abaqus 2024 documentation,
        """
        self.material.Elastic(table=((E.m, nu.m), ))
        self._elasticity = "Isotropic"

    def BilaminaElasticity(self, E_1plus: Quantity, E_2plus: Quantity, nu_12plus: Quantity, G_12: Quantity,
            E_1minus: Quantity, E_2minus: Quantity, nu_12minus: Quantity):
        """
        Define bilamina elasticity for the material.

        :param E_1plus: tensile Young's modulus in local 1-direction
        :param E_2plus: tensile Young's modulus in local 2-direction
        :param nu_12plus: tensile Poisson's ratio
        :param G_12: shear modulus
        :param E_1minus: compressive Young's modulus in local 1-direction
        :param E_2minus: compressive Young's modulus in local 2-direction
        :param nu_12minus: compressive Poisson's ratio

        :source
        [1] Abaqus 2024 documentation,
        https://docs.software.vt.edu/abaqusv2024/English/SIMACAEMATRefMap/simamat-c-linearelastic.htm#simamat-c-linearelastic-planestressbilamina
        """
        self.material.Elastic(
            table=((E_1plus.m, E_2plus.m, nu_12plus.m, G_12.m, E_1minus.m, E_2minus.m, nu_12minus.m), ), type=BILAMINA)
        self._elasticity = "Bilamina"
        self.info["elasticity"] = {"E_1plus": E_1plus.dict, "E_2plus": E_2plus.dict, "nu_12plus": nu_12plus.dict,
            "G_12": G_12.dict, "E_1minus": E_1minus.dict, "E_2minus": E_2minus.dict, "nu_12minus": nu_12minus.dict}

    def LaminaElasticity(self, E_1: Quantity, E_2: Quantity, nu_12: Quantity,
            G_12: Quantity, G_13: Quantity, G_23: Quantity):
        """
        Define lamina elasticity for the material.

        :param E_1: Young's modulus in local 1-direction
        :param E_2: Young's modulus in local 2-direction
        :param nu_12: Poisson's ratio
        :param G_12: shear modulus in plane 1-2
        :param G_13: shear modulus in plane 1-3
        :param G_23: shear modulus in plane 2-3

        :source
        [1] Abaqus 2024 documentation,
        https://docs.software.vt.edu/abaqusv2024/English/SIMACAEMATRefMap/simamat-c-linearelastic.htm#simamat-c-linearelastic-planestress
        """
        self.material.Elastic(table=((E_1.m, E_2.m, nu_12.m, G_12.m, G_13.m, G_23.m), ), type=LAMINA)
        self._elasticity = "Lamina"
        self.info["elasticity"] = {"E_1": E_1.dict, "E_2": E_2.dict, "nu_12": nu_12.dict, "G_12": G_12.dict,
            "G_13": G_13.dict, "G_23": G_23.dict}

    def EngineeringConstantsElasticity(self, E_1, E_2, E_3, nu_12, nu_13, nu_23, G_12, G_13, G_23):
        """
        Define engineering constants elasticity for the material.

        :param E_1: Young's modulus in local 1-direction
        :param E_2: Young's modulus in local 2-direction
        :param E_3: Young's modulus in local 3-direction
        :param nu_12: Poisson's ratio
        :param nu_13: Poisson's ratio
        :param nu_23: Poisson's ratio
        :param G_12: shear modulus
        :param G_13: shear modulus
        :param G_23: shear modulus

        :source
        [1] Abaqus 2024 documentation,
        https://docs.software.vt.edu/abaqusv2024/English/SIMACAEMATRefMap/simamat-c-linearelastic.htm#simamat-c-linearelastic-engconst
        """
        self.material.Elastic(table=((E_1.m, E_2.m, E_3.m, nu_12.m, nu_13.m, nu_23.m, G_12.m, G_13.m, G_23.m), ),
            type=ENGINEERING_CONSTANTS)
        self._elasticity = "Engineering Constants"
        self.info["elasticity"] = {"E_1": E_1.dict, "E_2": E_2.dict, "E_3": E_3.dict, "nu_12": nu_12.dict,
            "nu_13":nu_13.dict, "nu_23": nu_23.dict, "G_12": G_12.dict, "G_13": G_13.dict, "G_23": G_23.dict}

    def ElasticityFailStress(self, sigma_yield_2):
        self.material.elastic.FailStress(
            table=((sigma_yield_2.m, sigma_yield_2.m, sigma_yield_2.m, sigma_yield_2.m,
                sigma_yield_2.m, sigma_yield_2.m, sigma_yield_2.m), ))
        self.info["elasticity_fail_stress"] = {"sigma_yield_2": sigma_yield_2.dict}

    def HashinDamage(self, X_T: Quantity, X_C: Quantity, Y_T: Quantity, Y_C: Quantity, S_L: Quantity, S_T: Quantity,
            alpha: int = 0):
        """

        :param X_T: longitudinal tensile strength
        :param X_C: longitudinal compressive strength
        :param Y_T: transverse tensile strength
        :param Y_C: transverse compressive strength
        :param S_L: longitudinal shear strength
        :param S_T: transverse shear strength
        :param alpha: if set to 0, the fiber tension criterion depends only on X_T. If set to 1, the fiber tension
        criterion takes into account a term of S_L as well
        :return:
        """
        self.material.HashinDamageInitiation(table=((X_T.m, X_C.m, Y_T.m, Y_C.m, S_L.m, S_T.m), ), alpha=alpha)
        self._damage_initiation_criterion = "Hashin"
        self.info["hashin_damage"] = {"X_T": X_T.dict, "X_C": X_C.dict, "Y_T": Y_T.dict, "Y_C": Y_C.dict,
            "S_L": S_L.dict, "S_T": S_T.dict, "alpha": alpha}

    def HashinDamageEvolution(self, G_ftc: Quantity, G_fcc: Quantity, G_mtc: Quantity, G_mcc: Quantity):
        """

        :param G_ftc: energy dissipated during damage for fiber tension failure mode (units of enerergy per unit area)
        :param G_fcc: energy dissipated during damage for fiber compression failure mode (units of enerergy per unit area)
        :param G_mtc: energy dissipated during damage for matrix tension failure mode (units of enerergy per unit area)
        :param G_mcc: energy dissipated during damage for matrix compression failure mode (units of enerergy per unit area)
        :return:
        """
        if self._damage_initiation_criterion != "Hashin":
            raise ValueError("Hashin damage evolution can only be used if the damage initiation criterion is "
                             "also Hashin")
        self.material.hashinDamageInitiation.DamageEvolution(table=((G_ftc.m, G_fcc.m, G_mtc.m, G_mcc.m), ), type=ENERGY)
        self._damage_evolution = "Hashin"
        self.info["hashin_damage_evolution"] = {"G_ftc": G_ftc.dict, "G_fcc": G_fcc.dict,
            "G_mtc": G_mtc.dict, "G_mcc": G_mcc.dict}

    def HashinDamageStabilisation(self, eta_ft: float, eta_fc: float, eta_mt: float, eta_mc: float):
        """
        Define a viscous regularisation for the Hashin damage model.

        :param eta_ft: viscous regularisation coefficient for fiber tension failure mode
        :param eta_fc: viscous regularisation coefficient for fiber compression failure mode
        :param eta_mt: viscous regularisation coefficient for matrix tension failure mode
        :param eta_mc: viscous regularisation coefficient for matrix compression failure mode
        :return:
        """
        if self._damage_initiation_criterion != "Hashin" or self._damage_evolution != "Hashin":
            raise ValueError("Hashin damage stabilisation can only be used if the damage initiation criterion "
                             "is also Hashin and the damage evolution is also Hashin")
        self.material.hashinDamageInitiation.DamageStabilization(fiberTensileCoeff=eta_ft, fiberCompressiveCoeff=eta_fc,
            matrixTensileCoeff=eta_mt, matrixCompressiveCoeff=eta_mc)
        self.info["hashin_damage_stabilisation"] = {"eta_ft": eta_ft, "eta_fc": eta_fc,
            "eta_mt": eta_mt, "eta_mc": eta_mc}

    def PlyFabricDamage(self, X_1plus: Quantity | NDArray[Quantity], X_1minus: Quantity | NDArray[Quantity],
                        X_2plus: Quantity | NDArray[Quantity], X_2minus: Quantity | NDArray[Quantity],
                        S: Quantity | NDArray[Quantity], T: Quantity | NDArray[Quantity] | None = None, **kwargs):
        """
        Add Ply Fabric damage initiation criterion to the model. This is not possible in Abaqus CAE, but only by
        directly editing the .inp file.

        *Note: the Ply Fabric damage initiation criterion can only be used in Abaqus/Explicit.*

        For all parameters: these can be either a float (a single entry) or an array (in which case all input arrays
        need to have the same length). In case of an array, all first values will form one input line, all second
        values will form the second input line etc. (This allows to define temperature and field variable dependent
        criterions).

        :param X_1plus: tensile strength along fiber direction 1
        :param X_1minus: compressive strenght along fiber direction 1
        :param X_2plus: tensile strength along fiber direction 2
        :param X_2minus: compressive strength along fiber direction 2
        :param S: shear strength
        :param T: temperature, if temperature dependent
        :param kwargs: up to 2 other field variables - these can only be defined if all parameters are arrays and T
          is defined

        :source
        [1] Abaqus 2024 documentation,
        https://docs.software.vt.edu/abaqusv2024/English/?show=SIMACAEKEYRefMap/simakey-r-damageinitiation.htm
        """
        if self.model.type != "Explicit":
            raise ValueError("Ply Fabric damage initiation criterion can only be used in Abaqus/Explicit")

        if self._elasticity != "Bilamina":
            raise ValueError("Ply Fabric damage initiation criterion should only be used if the elasticity "
                             "is set to Bilamina")

        input_float = isinstance(X_1plus, Quantity)
        n_lines = None if input_float else len(X_1plus)

        n_field_vars = len(kwargs)

        if input_float and not (isinstance(X_1minus, Quantity) and isinstance(X_2plus, Quantity) and
                                isinstance(X_2minus, Quantity) and (isinstance(T, Quantity) or T is None)):
            raise ValueError("Either give all Quantity values or all array values")

        if not input_float and not (len(X_1minus) == n_lines and len(X_2plus) == n_lines and len(X_2minus) == n_lines and
                 len(S) == n_lines and len(T) == n_lines):
            raise ValueError("Either give all array values of the same length or all Quantity values")

        for field_var in kwargs.items():
            if input_float:
                raise ValueError("Field variables can only be defined if all parameters are arrays")
            elif T is None:
                raise ValueError("Field variables can only be defined if T is defined")
            elif len(field_var) != n_lines:
                raise ValueError("All field variable arrays need to have the same length as the other parameter arrays")

        field_vars = np.array(kwargs.items())

        # Input text
        text = "*Damage Initiation, criterion=PLY FABRIC\n"
        if input_float:
            text += f"{X_1plus.m}, {X_1minus.m}, {X_2plus.m}, {X_2minus.m}, {S.m}{', ' + T.m if T is not None else ''}\n"
        else:
            text += "\n".join(
                f"{X_1plus[i].m}, {X_1minus[i].m}, {X_2plus[i].m}, {X_2minus[i].m}, {S[i].m}, {T[i].m}{', ' + field_vars[0][i].m if n_field_vars >= 1 else ''}{', ' + field_vars[1][i].m if n_field_vars >= 2 else ''}" for i in range(n_lines)) + "\n"

        self._ply_fabric_initiation_number_of_lines = text.count("\n") - 1
        line_above = f"*Material, name={self.material.name}"
        self.model.actions_on_inp.append(lambda inp_file_path: insert_in_input_file(inp_file_path, text, line_above=line_above))

        self._damage_initiation_criterion = "Ply Fabric"

    def PlyFabricDamageEvolution(self, G_f1plus: Quantity, G_f1minus: Quantity, G_f2plus: Quantity, G_f2minus: Quantity, alpha_12max: Quantity, d_12: Quantity, T=None, **kwargs):
        """
        Add Ply Fabric damage evolution to the model. This is not possible in Abaqus CAE, but only by directly editing
        the .inp file.

        *Note: the Ply Fabric damage evolution can only be used in Abaqus/Explicit, and only if the damage initiation
        criterion is also Ply Fabric and the elasticity is set to bilamina.*

        *Note: for the moment not working: segfaults*

        :param G_f1plus: tensile fracture energy in local 1-direction (units of FL^-1)
        :param G_f1minus: compressive fracture energy in local 1-direction (units of FL^-1)
        :param G_f2plus: tensile fracture energy in local 2-direction (units of FL^-1)
        :param G_f2minus: compressive fracture energy in local 2-direction (units of FL^-1)
        :param alpha_12max: parameter for shear damage
        :param d_12: maximum shear damage
        :param T: temperature, if temperature dependent
        :param kwargs: field variables - maximum 5

        :return:

        :source
        [1] Abaqus 2024 documentation, https://docs.software.vt.edu/abaqusv2024/English/SIMACAEMATRefMap/simamat-c-linearelastic.htm
        and https://docs.software.vt.edu/abaqusv2024/English/SIMACAEKEYRefMap/simakey-r-damageevolution.htm#simakey-r-damageevolution__simakey-r-damageevolution-s-datadesc-plyfabric
        """
        tprint("Warning: Ply Fabric damage evolution is currently not working and causes Abaqus to segfault. Use with caution, and make sure to check the results carefully.", color="red")
        if self.model.type != "Explicit":
            raise ValueError("Ply Fabric damage evolution can only be used in Abaqus/Explicit")
        elif self._elasticity != "Bilamina":
            raise ValueError("Ply Fabric damage evolution can only be used if the elasticity is set to Bilamina")
        if self._damage_initiation_criterion != "Ply Fabric":
            raise ValueError("Ply Fabric damage evolution can only be used if the damage initiation criterion is "
                             "also Ply Fabric")

        input_float = isinstance(G_f1plus, Quantity)
        n_lines = None if input_float else len(G_f1plus)

        n_field_vars = len(kwargs)

        if input_float and not (isinstance(G_f1minus, Quantity) and isinstance(G_f2plus, Quantity) and isinstance(G_f2minus, Quantity) and isinstance(alpha_12max, Quantity) and isinstance(d_12, Quantity) and (isinstance(T, Quantity) or T is None)):
            raise ValueError("Either give all Quantity values or all array values")
        if not input_float and not (len(G_f1minus) == n_lines and len(G_f2plus) == n_lines and len(G_f2minus) == n_lines and len(alpha_12max) == n_lines and len(d_12) == n_lines and len(T) == n_lines):
            raise ValueError("Either give all array values of the same length or all Quantity values")

        for field_var in kwargs.items():
            if input_float:
                raise ValueError("Field variables can only be defined if all parameters are arrays")
            elif T is None:
                raise ValueError("Field variables can only be defined if T is defined")
            elif len(field_var) != n_lines:
                raise ValueError("All field variable arrays need to have the same length as the other parameter arrays")

        field_vars = np.array(kwargs.items())

        # Input text
        text = "*Damage Evolution, type=ENERGY, softening=EXPONENTIAL\n"
        if input_float:
            text += f"{G_f1plus.m}, {G_f1minus.m}, {G_f2plus.m}, {G_f2minus.m}, {alpha_12max.m}, {d_12.m}{', ' + T.m if T is not None else ''}\n"
        else:
            text += "\n".join(
                f"{G_f1plus[i].m}, {G_f1minus[i].m}, {G_f2plus[i].m}, {G_f2minus[i].m}, {alpha_12max[i].m}, {d_12[i].m}, {T[i].m}{', ' + field_vars[0][i].m if n_field_vars >= 1 else ''}{', ' + field_vars[1][i].m if n_field_vars >= 2 else ''}{', ' + field_vars[2][i].m if n_field_vars >= 3 else ''}{', ' + field_vars[3][i].m if n_field_vars >= 4 else ''}{', ' + field_vars[4][i].m if n_field_vars >= 5 else ''}" + "\n"
                for i in range(n_lines))

        line_above = f"*Damage Initiation, criterion=PLY FABRIC"
        self.model.actions_on_inp.append(lambda inp_file_path: insert_in_input_file(inp_file_path, text, line_above=line_above, offset=self._ply_fabric_initiation_number_of_lines))
        self._damage_evolution = "Ply Fabric"


class StainlessSteelMaterial(Material):
    def __init__(self, model: Model, name: str, description: str | None = None):
        super().__init__(model, name, description)
        self.Elasticity(E=200000 * ureg.MPa, nu=0.3 * ureg.dimensionless)
        self.Density(7850 * ureg.kg / ureg.m ** 3)

    def Plasticity(self):
        """Initially planned"""
        raise NotImplementedError()


def create_FRP_material_elastic(name: str, description: str = "", **kwargs) -> Callable[[Model], FRPMaterial]:
    """
    Create an FRP material with the given name and description, using the provided keyword arguments for
    changes from the default material properties. The model uses a simple isotropic elastic behaviour, without damage.

    :param name:
    :param description:
    :param kwargs:
    :return:
    """
    def material_factory(model: Model) -> FRPMaterial:
        material = FRPMaterial(model, name=name, description=description)
        material.Density(
            density=kwargs.get("Density", 1280 * ureg.kg / ureg.m ** 3)
        )
        material.IsotropicElasticity(
            E=kwargs.get("E", 38640 * ureg.MPa),
            nu=kwargs.get("nu", 0.3 * ureg.dimensionless)
        )
        material.ExpansionCoefficient(1 / ureg.degK)  # Because also used as wires
        return material
    return material_factory


def create_FRP_material_hashin(name: str, description: str = "", include_damage_evolution: bool = True,
        bidirectional: bool = False, **kwargs) -> Callable[[Model], FRPMaterial]:
    """
    Create an FRP material with the given name and description, using the provided keyword arguments for
    changes from the default material properties. The model uses the Hashin damage criterion and evolution.

    :param name:
    :param description:
    :param kwargs:
    :return:
    """
    def material_factory(model: Model) -> FRPMaterial:
        material = FRPMaterial(model, name=name, description=description)
        material.Density(
            density=kwargs.get("Density", 1280 * ureg.kg / ureg.m ** 3)
        )
        material.LaminaElasticity(
            E_1=kwargs.get("E_1", 38640 * ureg.MPa),  # Compressive test, initial E_C (lower than E_T)
            E_2=kwargs.get("E_2", 38640 * ureg.MPa),  # Compressive test, initial E_C (lower than E_T)
            nu_12=kwargs.get("nu_12", 0.3 * ureg.dimensionless),
            G_12=kwargs.get("G_12", 5100 * ureg.MPa),
            G_13=kwargs.get("G_13", 2000 * ureg.MPa),
            G_23=kwargs.get("G_23", 2000 * ureg.MPa)
        )
        material.HashinDamage(
            X_T=kwargs.get("X_T", 535.8 * ureg.MPa),  # Tensile test data
            X_C=kwargs.get("X_C", 389.9 * ureg.MPa),  # Compressive test data
            Y_T=kwargs.get("Y_T", 535.8 * ureg.MPa),  # Tensile test data (assumed = X_T)
            Y_C=kwargs.get("Y_C", 389.9 * ureg.MPa),  # Compressive test data (assumed = Y_T)
            S_L=kwargs.get("S_T", 70 * ureg.MPa),
            S_T=kwargs.get("S_C", 70 * ureg.MPa if not bidirectional else kwargs.get("Y_C", 389.9*ureg.MPa)/2),
            alpha=1
        )
        if include_damage_evolution:
            material.HashinDamageEvolution(
                G_ftc=kwargs.get("G_ftc", 27.5 * ureg.mJ / ureg.mm ** 2),  # Tensile test parameter study
                G_fcc=kwargs.get("G_fcc", 9 * ureg.mJ / ureg.mm ** 2),  # Compressive test parameter study
                G_mtc=kwargs.get("G_mtc", 27.5 * ureg.mJ / ureg.mm ** 2),  # Tensile test parameter study
                G_mcc=kwargs.get("G_mcc", 9 * ureg.mJ / ureg.mm ** 2)   # Compressive test parameter study
            )
            material.HashinDamageStabilisation(
                kwargs.get("eta_ft", 1e-7), kwargs.get("eta_fc", 1e-7), kwargs.get("eta_mt", 1e-7), kwargs.get("eta_mc", 1e-7)
            )
        return material
    return material_factory


def create_FRP_material_plyfabric(name: str, description: str = "", **kwargs) -> Callable[[Model], FRPMaterial]:
    """
    Create an FRP material with the given name and description, using the provided keyword arguments for
    changes from the default material properties. The model uses the Hashin damage criterion and evolution.

    :param name:
    :param description:
    :param kwargs:
    :return:
    """
    def material_factory(model: Model) -> FRPMaterial:
        material = FRPMaterial(model, name=name, description=description)
        material.Density(
            density=kwargs.get("Density", 1280 * ureg.kg / ureg.m ** 3)
        )
        material.BilaminaElasticity(
            E_1plus=kwargs.get("E_1plus", 33223 * ureg.MPa),
            E_2plus=kwargs.get("E_2plus", 33223 * ureg.MPa),
            nu_12plus=kwargs.get("nu_12plus", 0.3 * ureg.dimensionless),
            G_12=kwargs.get("G_12", 5100 * ureg.MPa),
            E_1minus=kwargs.get("E_1minus", 33223 * ureg.MPa),
            E_2minus=kwargs.get("E_2minus", 33223 * ureg.MPa),
            nu_12minus=kwargs.get("nu_12minus", 0.3 * ureg.dimensionless)
        )
        material.PlyFabricDamage(
            X_1plus=kwargs.get("X_1plus", 550 * ureg.MPa),
            X_2plus=kwargs.get("X_2plus", 550 * ureg.MPa),
            X_1minus=kwargs.get("X_1minus", 375 * ureg.MPa),
            X_2minus=kwargs.get("X_2minus", 375 * ureg.MPa),
            S=kwargs.get("S", 70 * ureg.MPa)
        )
        material.PlyFabricDamageEvolution(
            G_f1plus=kwargs.get("G_f1plus", 60 * ureg.mJ / ureg.mm ** 2),
            G_f1minus=kwargs.get("G_f1minus", 30 * ureg.mJ / ureg.mm ** 2),
            G_f2plus=kwargs.get("G_f2plus", 60 * ureg.mJ / ureg.mm ** 2),
            G_f2minus=kwargs.get("G_f2minus", 30 * ureg.mJ / ureg.mm ** 2),
            alpha_12max=kwargs.get("alpha_12max", 0.5 * ureg.dimensionless),
            d_12=kwargs.get("d_12", 0.5 * ureg.dimensionless)
        )
        return material

    return material_factory


def create_NiTiCo_material(name: str, description: str = "", **kwargs) -> Callable[[Model], SMAMaterial]:
    """
    Create a NiTiCo SMA material with the given name and description, using the provided keyword arguments for
    changes from the default material properties. The model uses the built-in Abaqus SMA material model.

    :param name:
    :param description:
    :param kwargs:
    :return:
    """
    def material_factory(model: Model) -> SMAMaterial:
        material = SMAMaterial(model, name=name, description=description)
        material.Density(
            density=kwargs.get("Density", 6450 * ureg.kg / ureg.m ** 3)
        )
        material.ExpansionCoefficient(1 / ureg.degK)
        material.Elasticity(
            E=kwargs.get("E", 34197 * ureg.MPa),
            nu=kwargs.get("nu", 0.33 * ureg.dimensionless)
        )
        material.SuperElasticity(
            E_M=kwargs.get("E_m", 20334 * ureg.MPa),
            nu_M=kwargs.get("nu_m", 0.33 * ureg.dimensionless),
            epsilon_L=kwargs.get("epsilon_L", 0.0154 * ureg.dimensionless),
            sigma_tL_S=kwargs.get("sigma_tL_S", 858 * ureg.MPa),
            sigma_tL_E=kwargs.get("sigma_tL_E", 887 * ureg.MPa),
            sigma_tU_S=kwargs.get("sigma_tU_S", 660 * ureg.MPa),
            sigma_tU_E=kwargs.get("sigma_tU_E", 645 * ureg.MPa),
            sigma_cL_S=kwargs.get("sigma_cL_S", 810 * ureg.MPa)
        )
        material.SuperElasticityHardening(
            yield_stress=kwargs.get("yield_stress", [896, 1146, 1381, 1753, 2046] * ureg.MPa),
            yield_strain=kwargs.get("yield_strain", [0.0625, 0.0875, 0.1125, 0.1375, 0.1625] * ureg.dimensionless),
        )
        return material
    return material_factory


def create_AISI302_material(name: str, description: str = "", **kwargs) -> Callable[[Model], StainlessSteelMaterial]:
    """
    Create an AISI 302 stainless steel material with the given name and description, using the provided keyword arguments for
    changes from the default material properties. The model uses the built-in Abaqus stainless steel material model.

    :param name:
    :param description:
    :param kwargs:
    :return:
    """
    def material_factory(model: Model) -> StainlessSteelMaterial:
        material = StainlessSteelMaterial(model, name=name, description=description)
        material.Density(
            density=kwargs.get("Density", 7920 * ureg.kg / ureg.m ** 3)  # From Goodfellow datasheet
        )
        material.ExpansionCoefficient(1 / ureg.degK)  # For practical purpose
        material.Elasticity(
            E=kwargs.get("E", 44100 * ureg.MPa),  # Mean from tests (VERY HIGH)
            nu=kwargs.get("nu", 0.27 * ureg.dimensionless)  # https://www.modulusmetal.com/aisi-302-stainless-steel-material-data-sheet/
        )
        # material.Plasticity()
        return material
    return material_factory
