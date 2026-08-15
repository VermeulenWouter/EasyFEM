"""
@date: 2026-05-22
@author: Wouter Vermeulen

Version 2
* Refactor code to be more modular and reusable

Code to generate a complete Abaqus model of a 3-point bending, tension or compressive test of an FRP-SMA Hybrid
Composite sample, with the possibility to use symmetry to reduce model size, and to increase mesh density.

The model can be viewed and checked in Abaqus/Viewer after running the scripts, as the models can be saved as
``.cae`` files!

External references:
  * Creating materials can be done with the functions from ``utils.abaqus.materials``
  * Running the model can be done with the functions from ``utils.abaqus``
  * Extracting results can be done with the functions from ``utils.abaqus.postprocessing``
  * Input values are mostly ``Quantity`` from ``utils.units``
"""
from typing import Callable

from abaqusConstants import *
from regionToolset import Region
from mesh import ElemType

from dataclasses import dataclass
from enum import Enum, auto
import numpy as np

from utils.abaqus.materials import FRPMaterial, SMAMaterial, StainlessSteelMaterial, Material
from utils.abaqus.postprocessing import *
from utils.units import QuantityArray


def wire_number_from_percentage(wire_percentage: Quantity, wire_diameter: Quantity,
        sample_width: Quantity, sample_thickness: Quantity, ensure_even: bool = False,
        ensure_div_4: bool = False) -> tuple[int, Quantity]:
    """
    Calculate the number of wires needed to achieve a certain volume percentage of wires in the sample, given the
    sample dimensions and wire diameter. The function can ensure the number of wires is even or divisible by 4 for
    symmetry purposes. It will, however, always return the TOTAL number of wires in the full model, not the number
    of wires in one line or a half symmetry of the model.

    :param wire_percentage: the desired volume percentage of wires in the sample (e.g. 0.02*ureg.dimensionless or
        2*ureg.percent for 2%)
    :param sample_width: the width of the sample (e.g. 15*ureg.mm)
    :param sample_thickness: the thickness of the sample (e.g. 2*ureg.mm)
    :param wire_diameter: the diameter of the wires (e.g. 0.125*ureg.mm)
    :param ensure_even: whether to ensure the number of wires is even (e.g. to ensure symmetry in a half model)
    :param ensure_div_4: whether to ensure the number of wires is divisible by 4 (e.g. to ensure symmetry a half model
        with two rows of wires). If specified, no need to also specify ``ensure_even``.

    :return: the total number of wires needed in the full model to achieve the desired wire percentage, ensuring
         divisibility if specified;
        and the actual volume percentage of wires achieved with that number of wires (which may be slightly higher
         or lower than the desired percentage due to symmetry constraints)
    """
    if wire_percentage.m == 0:
        return 0, 0 * ureg.percent

    wire_percentage.ito(ureg.dimensionless)  # Convert percentage to float if not already dimensionless

    sample_section_area = sample_width * sample_thickness
    single_wire_section_area = np.pi * (wire_diameter / 2) ** 2
    n_wires = int((wire_percentage * sample_section_area / single_wire_section_area).m)

    if ensure_even:
        n_wires = n_wires if n_wires % 2 == 0 else n_wires + 1  # Ensure n_wires is even for single symmetry
    elif ensure_div_4:
        n_wires = n_wires + (4 - n_wires % 4) % 4  # Ensure n_wires is even and divisible by 4 for symmetry + two rows
    return n_wires, (n_wires * single_wire_section_area / sample_section_area).ito(ureg.percent)


def flatten_coords_tuple(coords: tuple) -> tuple[tuple[Quantity, Quantity, Quantity]]:
    """Flatten a tuple of tuples of coordinates into a single tuple of coordinates. This is needed to allow for both
    single and multiple coordinates to be specified in the ``find_geometry_by_coords`` function."""
    flattened = []
    for coord in coords:
        if isinstance(coord, tuple) and len(coord) == 3 and all(isinstance(c, (Quantity, float)) for c in coord):
            flattened.append(coord)
        elif isinstance(coord, tuple):
            flattened.extend(flatten_coords_tuple(coord))
        else:
            raise ValueError(f"Invalid coordinate format: {coord}. Must be a tuple of three Quantity objects "
                             f"(y, z, x) or a tuple of such tuples.")
    return tuple(flattened)


class GeometryType(Enum):
    EDGE = auto()
    CELL = auto()
    FACE = auto()
    VERTEX = auto()


def find_geometry_by_coords(part_or_instance, coords: tuple[tuple[Quantity, Quantity, Quantity]] | tuple,
        geometry_type: GeometryType = GeometryType.EDGE, expect_single: bool = False,
        transform: Callable[[tuple[Quantity, Quantity, Quantity]], tuple[Quantity, Quantity, Quantity]] | None = None):
    """
    Find the geometry (edge, face or cell) in the part or instance that corresponds to the specified coordinates.

    :param part_or_instance: the part or instance in which to search for the geometry. This can be either a Part object
        (if the geometry is not yet added to the assembly) or an Instance object (if the geometry is already added to
        the assembly).
    :param coords: the coordinates of the geometry to find. This can be a single tuple of (y, z, x) coordinates (e.g.
        for an edge or face) or a tuple of tuples of coordinates (e.g. for multiple edges or faces). While not
        strictly necessary, for legibility it is recommended to specify the coordinates at the centre of the geometry
        to be found.
    :param geometry_type: the type of geometry to find, which can be EDGE, FACE or CELL. This will determine whether
        the function searches for edges, faces or cells in the part or instance that correspond to the specified
        coordinates.
    :param expect_single: whether to expect a single geometry to be found at the specified coordinates. If True,
        the function will return the single geometry found at the specified coordinates, and will raise an error if
         multiple geometries are found.
    :param transform: an optional function to apply to the coordinates before searching for the geometry.
        This can be used e.g. if an instance is rotated in the assembly compared to the original part, to ensure the
        coordinates are correctly transformed.

    :return: the result of the ``findAt()`` method for the specified geometry type, which can be used to create sets
        or apply boundary conditions
    """
    if isinstance(coords[0], Quantity):
        coords = (coords,)
    else:
        coords = flatten_coords_tuple(coords)

    if transform is None:
        transform = lambda x: x

    coords_input = []
    for coord in coords:
        if not isinstance(coord, tuple) or len(coord) != 3:
            raise ValueError(f"Each coordinate must be a tuple of three Quantity objects (y, z, x), but got {coord}.")
        if all(isinstance(c, float) for c in coord):
            coords_input.append((coord,))
        else:
            coords_input.append((tuple(c.m for c in transform(coord)),))

    if geometry_type == GeometryType.EDGE:
        result = part_or_instance.edges.findAt(*coords_input)
    elif geometry_type == GeometryType.FACE:
        result = part_or_instance.faces.findAt(*coords_input)
    elif geometry_type == GeometryType.CELL:
        result = part_or_instance.cells.findAt(*coords_input)
    elif geometry_type == GeometryType.VERTEX:
        result = part_or_instance.vertices.findAt(*coords_input)
    else:
        raise ValueError(f"Invalid geometry type {geometry_type}. Must be EDGE, FACE or CELL.")

    if not expect_single:
        return result
    else:
        if len(result) == 1:
            return result[0]
        else:
            raise ValueError(f"Expected a single geometry to be found at the specified coordinates, but found "
                             f"{len(result)} geometries.")


class MeshType(Enum):
    COARSE = auto()
    TRANSITION = auto()
    FINE = auto()


@dataclass
class CellProperties:
    # Geometric information
    x_start: Quantity
    x_end: Quantity
    y_mid: Quantity
    z_mid: Quantity

    # Abaqus entities
    part: object

    mesh_type: MeshType = MeshType.COARSE

    # Optional mesh metadata
    sweep_edge_coord: tuple | None = None
    bias_direction: int | None = None

    @property
    def x_mid(self) -> Quantity:
        return (self.x_start + self.x_end) / 2

    @property
    def cell_mid(self):
        """Return the coordinates of the middle point of the cell."""
        return self.y_mid, self.z_mid, self.x_mid

    @property
    def edges_in_thickness(self):
        """Return the coordinates of the middle points of the edges along the thickness direction (z) of the cell."""
        return tuple((y, self.z_mid, x) for y in (0*ureg.mm, self.y_mid*2) for x in (self.x_start, self.x_end))

    @property
    def edges_in_length(self):
        """Return the coordinates of the middle points of the edges along the length direction (x) of the cell."""
        return tuple((y, z, self.x_mid) for y in (0*ureg.mm, self.y_mid*2) for z in (0*ureg.mm, self.z_mid*2))

    @property
    def edges_in_width(self):
        """Return the coordinates of the middle points of the edges along the width direction (y) of the cell."""
        return tuple((self.y_mid, z, x) for x in (self.x_start, self.x_end) for z in (0*ureg.mm, self.z_mid*2))

    @property
    def top_face(self):
        """Return the coordinates of the middle point of the face on the top of the cell (the face in the
        (length, width) plane)."""
        return self.y_mid, self.z_mid*2, self.x_mid


class TestType(Enum):
    """Overview of implemented test types"""
    THREE_POINT_BENDING = {"Name": "3-point bending", "Orientation": "Horizontal",
        "Supports": "Edge", "Loading": "Edge",
        "Wire": "Multiple",
        "Symmetry": "Lateral and/or longitudinal", "Sample": "FRP or FRP-wire hybrid"}
    TENSION = {"Name": "Tension", "Orientation": "Vertical", "Supports": "Face", "Loading": "Face", "Wire": "Multiple",
        "Symmetry": None, "Sample": "FRP or FRP-wire hybrid"}
    WIRE_TENSION = {"Name": "Wire tension", "Orientation": "Vertical", "Supports": "Point", "Loading": "Point",
        "Wire": "Single", "Symmetry": None, "Sample": "Wire only"}
    COMPRESSION = {"Name": "Compression", "Orientation": "Vertical", "Supports": "Face", "Loading": "Face",
        "Wire": "Multiple", "Symmetry": None, "Sample": "FRP or FRP-wire hybrid"}
    DYNAMIC = {"Name": "Dynamic", "Orientation": "Vertical", "Supports": "Face", "Loading": None, "Wire": "Multiple",
        "Symmetry": None, "Sample": "FRP or FRP-wire hybrid"}


class FRPSample:
    def __init__(self, model, test_type: TestType, symmetry: str | None = "lateral+longitudinal") -> None:
        """

        :param model: the Abaqus model to which the part will be added. This is needed to create the part, assign
            the section, ...
        :param test_type: the type of test to model.
            This will determine how the support and loading regions are created. It also changes the orientation of
            the sample (either horizontal or vertical).
        :param symmetry: ``lateral`` (lateral symmetry = model only half width), ``longitudinal``
            (longitudinal symmetry = model only half the length) or ``lateral+longitudinal`` (both symmetries)
            In function of this, the geometry will be created with the corresponding dimensions (e.g. the ``width``
            will be halved if the model is symmetric laterally). The symmetry boundary conditions will also be
            automatically created when the part is added to the assembly.
        """
        self.__part_name = "FRP Sample"

        self.__model = model
        self.__model_type = model.type
        self.test_type = test_type

        self.part = None
        self.instance = None
        self.local_csys = None
        self._layup_name = None
        self.set_frp_sample = None
        self.set_supports = None
        self.set_loading = None

        self.__cell_properties = []
        self.__cells = None

        self.__coord_regions_supports = None
        self.__coord_regions_loading = None

        # Symmetry
        self.__symmetry_lateral = "lateral" in symmetry if symmetry else False
        self.__symmetry_longitudinal = "longitudinal" in symmetry if symmetry else False
        self.__coord_faces_symmetry_lateral = None
        self.__coord_faces_symmetry_longitudinal = None

        self.info = {
            "Symmetry": symmetry,
        }

        # Control variables to avoid forgetting a step or doing things in the wrong order
        self.__geometry_created = False
        self.__section_assigned = False
        self.__added_to_assembly = False
        self.__meshed = False

    def create_geometry(self, width: Quantity, length: Quantity, thickness: Quantity,
            partitions: list[Quantity] | None = None,
            partition_support_line: Quantity | QuantityArray | None = None) -> None:
        """
        Create the geometry of the FRP sample part, which is a rectangular block with the specified dimensions.
        The part is created as a 3D deformable body, and a local coordinate system is defined aligned with the part
        (x along length, y along width, z along thickness). The sample is by default lying in the xz plane
        (positioned horizontally). Depending on the ``test_type`` defined when initialising this class, it will later
        be rotated when importing it into the assembly using ``add_to_assembly``.

        The part is partitioned at the specified distances from one end of the sample (e.g. to facilitate meshing
        and/or application of boundary).

        *Implementation note: to ensure legibility with coordinates, each ``findAt()`` is used to get to the middle
        point of whatever is searched for (e.g. centre point of an edge or a cell).

        Coordinates are specified in the direction (width, height, length) of the sample (this corresponds to the
        global coordinate system!). One of the corners of the sample will lie at
        (0,0,0) while all other will have only positive coordinates.*

        :param width: the width of the sample (e.g. 15*ureg.mm). If the model is symmetric, this should still be
            the full width.
        :param length: the length of the sample (e.g. 100*ureg.mm). If the model is symmetric, this should still be
            the full length.
        :param thickness: the thickness of the sample (e.g. 2*ureg.mm).
        :param partitions: list of distances from one end of the sample where a partition should be created (to
            facilitate meshing and/or BC application). All values should be between ``0`` and ``length``. If no
            partitions are needed, set to None. If the model is symmetric, values should be between ``0`` and
            ``length/2``.
        :param partition_support_line: for a 3-point bending test, the position of the support line(s) must be
            specified to create the corresponding partition
        """
        width = width if not self.__symmetry_lateral else width / 2
        length = length if not self.__symmetry_longitudinal else length / 2

        ##
        # Create 3D deformable body geometry
        ##
        sketch = self.__model.ConstrainedSketch(name="__profile__", sheetSize=200.0)
        sketch.rectangle(point1=(0.0, 0.0), point2=(width.m, thickness.m))

        # Add and position dimension arrows and text
        sketch.ObliqueDimension(textPoint=((width/2).m, thickness.m + 3), value=width.m,
            vertex1=sketch.vertices.findAt((0.0, thickness.m), ),
            vertex2=sketch.vertices.findAt((width.m, thickness.m), ))
        sketch.ObliqueDimension(textPoint=(-3, (thickness/2).m), value=thickness.m,
            vertex1=sketch.vertices.findAt((0.0, 0.0), ),
            vertex2=sketch.vertices.findAt((0.0, thickness.m), ))

        # Extrude part
        self.part = self.__model.Part(dimensionality=THREE_D, name=self.__part_name, type=DEFORMABLE_BODY)
        self.part.BaseSolidExtrude(depth=length.m, sketch=sketch)

        del sketch

        ##
        # Create coordinate system aligned with the part (x along length, y along width, z along thickness)
        ##
        part_coord_system_feature = self.part.DatumCsysByThreePoints(coordSysType=CARTESIAN, name="FRP_Local",
            origin=(0.0, 0.0, 0.0), point1=(0.0, 0.0, 1.0), point2=(1.0, 0.0, 0.0))

        self.local_csys = self.part.datums[part_coord_system_feature.id]

        ##
        # Create partitions
        ##
        partitions = sorted(partitions or [])
        for partition in partitions:
            if not (0 < partition.m < length.m):
                raise ValueError(f"Partition value {partition:~P} is out of bounds.")
            plane_feature = self.part.DatumPlaneByPrincipalPlane(offset=partition.m, principalPlane=XYPLANE)
            datum = self.part.datums[plane_feature.id]
            self.part.PartitionCellByDatumPlane(cells=self.part.cells, datumPlane=datum)

        partition_positions_x = quantity_array([0*ureg.mm, *partitions, length])
        self.part.regenerate()

        ##
        # Create coordinate groups and cell topologies for later use
        ##
        for i in range(len(partition_positions_x)-1):
            cell_properties = CellProperties(x_start=partition_positions_x[i], x_end=partition_positions_x[i+1],
                y_mid=width/2, z_mid=thickness/2, part=self.part)
            self.__cell_properties.append(cell_properties)

        self.__cells = find_geometry_by_coords(
            self.part, tuple(cellprop.cell_mid for cellprop in self.__cell_properties), geometry_type=GeometryType.CELL)

        # Symmetry
        # Faces in the (length, thickness) plane
        if self.__symmetry_lateral:
            self.__coord_faces_symmetry_lateral = tuple((width, (thickness / 2), cellprop.x_mid) for
                cellprop in self.__cell_properties)

        # Faces in the (width, thickness) plane
        if self.__symmetry_longitudinal:
            self.__coord_faces_symmetry_longitudinal = (width / 2), (thickness / 2), length

        # Loading and support regions
        if self.test_type.value["Orientation"] == "Horizontal":
            if partition_support_line is None:
                raise ValueError("For a 3-point bending test, the 'partition_support_line' argument must be specified "
                                 "to indicate where the support line(s) should be created.")
            if self.__symmetry_longitudinal:
                self.__coord_regions_supports = ((width/2), 0.0*ureg.mm, partition_support_line)
                self.__coord_regions_loading = ((width/2), thickness, length)
                self.info["L0"] = (length - partition_support_line).dict
            else:
                self.__coord_regions_supports = tuple(((width/2), 0.0*ureg.mm, x) for x in partition_support_line)
                self.__coord_regions_loading = ((width/2), thickness, (length/2))
        elif self.test_type.value["Orientation"] == "Vertical":
            if self.test_type.value["Supports"] is not None:
                self.__coord_regions_supports = ((width/2), (thickness/2), 0.0*ureg.mm)
            if self.test_type.value["Loading"] is not None:
                self.__coord_regions_loading = ((width/2), (thickness/2), length)
        self.__geometry_created = True

        self.info["Width"] = width.dict
        self.info["Length"] = length.dict
        self.info["Thickness"] = thickness.dict

    def assign_composite_layup_section(self, material: FRPMaterial, **kwargs) -> None:
        """
        Create a composite layup section with the specified material and assign it to the part. The layup is created
        with the specified number of plies, and the plies are defined with the specified material. All plies are
        oriented in local direction 1 of the part (along the length of the sample), and the stacking direction is
        along local direction 3 (through the thickness of the sample).

        :param material: the material to use for the composite layup plies. Note direction 1 is in the length,
            direction 2 in the width and direction 3 in the thickness. Stacking direction is direction 3.
        :param kwargs:
            ``n_plies``: the number of plies to use in the composite layup. Default is 8 (based on experimental samples)
            ``integration``: the integration rule to use for the composite section. Default is ``SIMPSON``,
                can be ``GAUSS``.
            ``num_int_points_per_ply``: the number of integration points to use per ply. Default is 3.

        :return:
        """
        if not self.__geometry_created:
            raise RuntimeError("Geometry must be created before assigning composite layup section.")

        integration = kwargs.get("integration", SIMPSON)  # Should not be necessary to change this value
        num_int_points_per_ply = kwargs.get("num_int_points_per_ply", 3)  # Should not be necessary to change this value

        n_plies = kwargs.get("n_plies", 8)  # Yvan used 8 plies for 2mm thickness

        self._layup_name = f"Composite layup - {n_plies} plies"
        layup = self.part.CompositeLayup(name=self._layup_name,
            elementType=CONTINUUM_SHELL, symmetric=False)
        layup.Section(integrationRule=integration)
        layup.ReferenceOrientation(additionalRotationField="", additionalRotationType=ROTATION_NONE, angle=0.0,
            axis=AXIS_3, fieldName="", localCsys=self.local_csys, orientationType=SYSTEM,
            stackDirection=STACK_ORIENTATION)

        layup.suppress()
        for i in range(n_plies):
            layup.CompositePly(additionalRotationField="", additionalRotationType=ROTATION_NONE, angle=0.0,
                axis=AXIS_3, material=material.name, numIntPoints=num_int_points_per_ply, orientationType=ANGLE_0,
                plyName=f"Ply-{i+1}", region=Region(cells=self.part.cells),
                suppressed=False, thickness=1, thicknessType=SPECIFY_THICKNESS)  # Thickness is relative, not absolute!
        layup.resume()

        self.__section_assigned = True

        self.info["Number of plies"] = n_plies

    def __process_cell_mesh_type(self, fine_mesh_cells: list[Quantity]):
        """Determine the mesh type (coarse, fine or transition) for each cell based on the specified list of cells
        to be meshed finely."""
        for i, cellprop in enumerate(self.__cell_properties):
            for fine_mesh_cell in fine_mesh_cells:
                if cellprop.x_start < fine_mesh_cell < cellprop.x_end:
                    cellprop.mesh_type = MeshType.FINE

        for i, cellprop in enumerate(self.__cell_properties):
            if cellprop.mesh_type == MeshType.COARSE:
                if (i > 0 and self.__cell_properties[i-1].mesh_type == MeshType.FINE) or (
                        i < len(self.__cell_properties) - 1 and self.__cell_properties[i+1].mesh_type == MeshType.FINE):
                    if cellprop.mesh_type != MeshType.FINE:  # Don't overwrite if already set to fine
                        cellprop.mesh_type = MeshType.TRANSITION

    def mesh_part(self, coarse_mesh_size: Quantity, fine_mesh_size: Quantity = None,
            fine_mesh_cells: list[Quantity] | None = None, **kwargs) -> None:
        """
        Mesh the part with continuum shell elements (SC8R) and a structured mesh. If ``fine_mesh_size`` is specified
        (a partition must be indicated already in the ``create_geometry`` method for the finer mesh zone), a finer
        mesh will be created in these zones, and a zone with bias will be created between the finer and coarser mesh
        zones to ensure a smooth transition between them.

        :param coarse_mesh_size: the desired mesh size
        :param fine_mesh_size: the desired mesh size in the cells indicated in the ``create_geometry`` method
        :param fine_mesh_cells: a list of x coordinates indicating the cells in which a finer mesh should be created.
        :param kwargs:
            ``elements_in_thickness``: the number of elements through the thickness. Since composite layup sections are
                used, only 1 element through the thickness is expected (and this is what Yvan used in his models),
                but if needed this can be changed by specifying a different value here. It is, however, against abaqus
                documentation.
        :return:
        """
        if not self.__geometry_created:
            raise RuntimeError("Geometry must be created before meshing.")

        # Composite layup are expected to have a single element through their thickness
        # See https://docs.software.vt.edu/abaqusv2024/English/?show=SIMACAECAERefMap/simacae-m-PrpCompositesShellContinuum-sb.htm
        elements_in_thickness = kwargs.get("elements_in_thickness", 1)

        if fine_mesh_cells is not None:
            self.__process_cell_mesh_type(fine_mesh_cells)

        ##
        # Seed edges
        ##
        edges_in_thickness = find_geometry_by_coords(self.part, tuple(edge for edge in (edges for edges in
            (cellprop.edges_in_thickness for cellprop in self.__cell_properties))))
        self.part.seedEdgeByNumber(edges=edges_in_thickness, number=elements_in_thickness, constraint=FINER)

        self.part.assignStackDirection(cells=self.part.cells, referenceRegion=find_geometry_by_coords(
            self.part, self.__cell_properties[0].top_face, geometry_type=GeometryType.FACE, expect_single=True))

        for i, cellprop in enumerate(self.__cell_properties):
            # Ensure 'bottom' and 'top' are correctly identified

            if cellprop.mesh_type == MeshType.FINE:
                self.part.seedEdgeBySize(edges=find_geometry_by_coords(self.part, cellprop.edges_in_length),
                    deviationFactor=0.1, minSizeFactor=0.1, size=fine_mesh_size.m)
                self.part.seedEdgeBySize(edges=find_geometry_by_coords(self.part, cellprop.edges_in_width),
                    deviationFactor=0.1, minSizeFactor=0.1, size=fine_mesh_size.m)
            elif cellprop.mesh_type == MeshType.TRANSITION:
                if i == 0:
                    bias_direction = "end1Edges"
                elif i == len(self.__cell_properties) - 1:
                    bias_direction = "end2Edges"
                else:
                    bias_direction = "end1Edges" if (
                            self.__cell_properties[i-1].mesh_type == MeshType.FINE) else "end2Edges"

                self.part.seedEdgeByBias(biasMethod=SINGLE, constraint=FINER, maxSize=coarse_mesh_size.m,
                    minSize=fine_mesh_size.m,
                    **{bias_direction: find_geometry_by_coords(self.part, cellprop.edges_in_length)})
                self.part.setSweepPath(
                    edge=find_geometry_by_coords(self.part, cellprop.edges_in_thickness[0], expect_single=True),
                    region=find_geometry_by_coords(self.part, cellprop.cell_mid, geometry_type=GeometryType.CELL)[0],
                    sense=FORWARD)
                self.part.setMeshControls(elemShape=HEX, technique=SWEEP, minTransition=OFF, algorithm=MEDIAL_AXIS,
                    regions=find_geometry_by_coords(self.part, cellprop.cell_mid, geometry_type=GeometryType.CELL))

        self.part.seedPart(deviationFactor=0.1, minSizeFactor=0.1, size=coarse_mesh_size.m)

        ##
        # Set element type to continuum shell
        ###
        if self.__model_type == "Standard":
            self.part.setElementType(regions=tuple(self.__cells), elemTypes=(ElemType(elemCode=SC8R,
                elemLibrary=STANDARD, secondOrderAccuracy=ON, hourglassControl=DEFAULT),))
        elif self.__model_type == "Explicit":
            self.part.setElementType(regions=tuple(self.__cells), elemTypes=(ElemType(elemCode=SC8R,
                elemLibrary=EXPLICIT, secondOrderAccuracy=OFF, hourglassControl=DEFAULT),))

        ##
        # Generate mesh (starting with regular zones, otherwise transition zone will mesh wrongly)
        ##
        self.part.generateMesh()

        self.__model.rootAssembly.regenerate()

        self.__meshed = True

        self.info["Coarse mesh size"] = coarse_mesh_size.dict
        if fine_mesh_cells is not None:
            self.info["Fine mesh size"] = fine_mesh_size.dict
            self.info["Fine mesh cells"] = [fine_mesh_cell.dict for fine_mesh_cell in fine_mesh_cells]

    def add_to_assembly(self, **kwargs) -> None:
        """
        Add the part to the assembly, creating an instance of the part in the root assembly. Depending on the
        specified symmetry of the model, symmetry boundary conditions will be automatically created on the
        corresponding faces of the part. Sets will also be created for the support and loading regions (as defined
        in the ``create_geometry`` method) and for the whole sample, to facilitate application of boundary conditions
        and loading in later steps.

        Depending on the test type, the part will be rotated upon import to ensure the sample is lying in the
        correct orientation (e.g. horizontal for 3-point bending, vertical for tension and compression).

        :param kwargs:
        :return:
        """
        if not self.__geometry_created:
            raise RuntimeError("Geometry must be created before adding part to assembly.")
        if not self.__section_assigned:
            raise RuntimeError("Composite layup section must be assigned before adding part to assembly.")
        if not self.__meshed:
            # Meshing uses some findAt with coordinates. If the assembly has a different orientation than the default
            # one, this might mess things up (not tested, but better safe than sorry).
            raise RuntimeError("Part must be meshed before adding to assembly.")

        rAssembly = self.__model.rootAssembly
        rAssembly.DatumCsysByDefault(CARTESIAN)
        self.instance = rAssembly.Instance(dependent=ON, name="FRP Sample", part=self.part)

        ##
        # Rotate if necessary
        ##
        if self.test_type.value["Orientation"] == "Vertical":
            rAssembly.rotate(angle=-90.0, axisDirection=(1.0, 0.0, 0.0), axisPoint=(0.0, 0.0, 0.0),
                instanceList=(self.__part_name, ))

        def rotate_coords(coords: tuple[Quantity, Quantity, Quantity]) -> tuple[Quantity, Quantity, Quantity]:
            """Rotate coordinates from the HORIZONTAL to VERTICAL following the rotation defined in the
            ``add_to_assembly`` method. Only rotate if the sample is VERTICAL."""
            if self.test_type.value["Orientation"] == "Horizontal":
                return coords
            y, z, x = coords
            return y, x, -z

        ##
        # Ensure symmetry boundary conditions are created if needed (depending on the specified symmetry of the model)
        ##
        if self.__symmetry_longitudinal:
            set_faces_symmetry_longitudinal = rAssembly.Set("LongitudinalSymmetry",
                faces=find_geometry_by_coords(self.instance, self.__coord_faces_symmetry_longitudinal,
                    geometry_type=GeometryType.FACE, transform=rotate_coords))
            if self.test_type.value["Orientation"] == "Horizontal":
                self.__model.ZsymmBC(createStepName="Initial", name="LongitudinalSymmetry",
                    region=set_faces_symmetry_longitudinal)
            else:
                self.__model.YsymmBC(createStepName="Initial", name="LongitudinalSymmetry",
                    region=set_faces_symmetry_longitudinal)

        if self.__symmetry_lateral:
            set_faces_symmetry_lateral = rAssembly.Set("LateralSymmetry",
                faces=find_geometry_by_coords(self.instance, self.__coord_faces_symmetry_lateral,
                    geometry_type=GeometryType.FACE, transform=rotate_coords))
            self.__model.XsymmBC(createStepName="Initial", name="LateralSymmetry", region=set_faces_symmetry_lateral)

        ##
        # Prepare sets for boundary conditions and loading (though their creation will depend on the testing type
        # and is for another function)
        ##
        if self.__coord_regions_supports:
            if self.test_type.value["Supports"] == "Face":
                self.set_supports = rAssembly.Set(name="SupportsSet", faces=find_geometry_by_coords(
                    self.instance, self.__coord_regions_supports, GeometryType.FACE, transform=rotate_coords))
            elif self.test_type.value["Supports"] == "Edge":
                self.set_supports = rAssembly.Set(name="SupportsSet",
                    edges=find_geometry_by_coords(self.instance, self.__coord_regions_supports,
                        transform=rotate_coords))
            else:
                raise ValueError(f"Unsupported support type {self.test_type.value['Supports']} for test type "
                                 f"{self.test_type.value['Name']} for an FRPSample.")
        if self.__coord_regions_loading:
            if self.test_type.value["Loading"] == "Face":
                self.set_loading = rAssembly.Set(name="LoadingSet", faces=find_geometry_by_coords(
                    self.instance, self.__coord_regions_loading, GeometryType.FACE, transform=rotate_coords))
            elif self.test_type.value["Loading"] == "Edge":
                self.set_loading = rAssembly.Set(name="LoadingSet",
                    edges=find_geometry_by_coords(self.instance, self.__coord_regions_loading, transform=rotate_coords))
            else:
                raise ValueError(f"Unsupported loading type {self.test_type.value['Loading']} for test type "
                                 f"{self.test_type.value['Name']} for an FRPSample.")

        self.set_frp_sample = rAssembly.Set(cells=self.instance.cells, name="FRPSampleSet")

        rAssembly.regenerate()

        self.__added_to_assembly = True

    def info(self) -> dict[str, dict | str | bool]:
        return self.info


class WiresSample:
    def __init__(self, model, test_type: TestType, symmetry: str | None = "lateral+longitudinal", **kwargs) -> None:
        self.set_all_wires = None
        self.set_wire = None
        self.__model = model
        self.test_type = test_type
        self.__symmetry_lateral = "lateral" in symmetry if symmetry else False
        self.__symmetry_longitudinal = "longitudinal" in symmetry if symmetry else False

        self.part = None
        self.instance = None

        self.info = {}

        # Control variables to avoid forgetting a step or doing things in the wrong order
        self.__geometry_created = False
        self.__section_assigned = False
        self.__added_to_assembly = False
        self.__meshed = False

    def create_geometry(self, length: Quantity) -> None:
        """
        Create the geometry of the wire sample part, which is a straight wire with the specified length. The part is
        created as a 3D deformable body. By default, it is positioned horizontally along the global x axisn, but
        depending on the test type, it will be rotated when using the ``add_to_assembly`` method.

        :param length: the total length of the wire (e.g. 100*ureg.mm). If the model is symmetric, this should still
            be the full length.
        :return:
        """
        length = length if not self.__symmetry_longitudinal else length / 2
        sketch = self.__model.ConstrainedSketch(name='__profile__', sheetSize=200.0)
        sketch.Line(point1=(0.0, 0.0), point2=(length.m, 0.0))

        self.part = self.__model.Part(dimensionality=THREE_D, name="Wire", type=DEFORMABLE_BODY)
        self.part.BaseWire(sketch=sketch)
        del sketch

        # Ensure there always is a node at mid-length (for data extraction)
        self.part.PartitionEdgeByPoint(edge=self.part.edges[0],
            point=self.part.InterestingPoint(self.part.edges[0], MIDDLE))

        self.set_wire = self.part.Set(edges=self.part.edges, name="WireSet")

        self.__geometry_created = True

    def assign_section(self, diameter: Quantity, material: SMAMaterial | StainlessSteelMaterial | Material) -> None:
        """
        Create a circular section with the specified diameter and material, and assign it to the part.

        :param diameter: the diameter of the wire (e.g. 0.5*ureg.mm)
        :param material: the material to use for the wire section.
        :return:
        """
        if not self.__geometry_created:
            raise RuntimeError("Geometry must be created before assigning section.")

        section = self.__model.TrussSection(area=(diameter ** 2 / 4 * np.pi).m, material=material.name,
            name="WireSection")
        self.part.SectionAssignment(region=self.set_wire, sectionName=section.name)
        self.info["Diameter"] = diameter.dict

    def mesh_part(self, mesh_size: Quantity) -> None:
        """
        Mesh the part with linear beam elements (T3D2).

        :param mesh_size: the desired mesh size (e.g. 1*ureg.mm)
        """
        if not self.__geometry_created:
            raise RuntimeError("Geometry must be created before meshing.")

        self.part.seedPart(deviationFactor=0.1, minSizeFactor=0.1, size=mesh_size.m)
        if self.__model.type == "Standard":
            self.part.setElementType(elemTypes=(ElemType(elemCode=T3D2, elemLibrary=STANDARD), ),
                                regions=(self.part.edges,))
        elif self.__model.type == "Explicit":
            self.part.setElementType(elemTypes=(ElemType(elemCode=T3D2, elemLibrary=EXPLICIT), ),
                                regions=(self.part.edges,))
        self.part.generateMesh()

        self.info["Mesh size"] = mesh_size.dict

    def add_to_assembly(self, n_wires: int = 1, n_lines: int = 1, wire_offset_from_side: Quantity = 0*ureg.mm,
            wire_offset_from_top: Quantity = 0*ureg.mm, wire_interdistance_thickness: Quantity = 1*ureg.mm,
            wire_interdistance_width: Quantity = 1*ureg.mm) -> None:
        """
        Add the part to the assembly, creating an instance of the part in the root assembly. Depending on the
        specified test type, the part will be rotated upon import to ensure the sample is lying in
        the correct orientation (e.g. horizontal for 3-point bending, vertical for tension and compression).

        If multiple wires are specified, they will be distributed in a grid pattern with the specified interdistance
        in the thickness and width directions.

        *Note: this function divides the number of wires by 2 if lateral symmetry is specified! Other than that,
        it has no idea of the sample dimensions, so wires could be created outside of a possible FRP host.*

        :param n_wires: number of wires to add in the assembly. If more than 1, all other parameters related to the
            distribution of the wires must be specified. Default is 1 (i.e. a single wire).
        :param n_lines: number of lines in which the wires will be distributed (a line = a row of wires in the width
            direction). Thus, this specifies how many 'layers' of wires should be placed in the thickness direction.
        :param wire_offset_from_side: the distance of the first wires from the side of the sample
        :param wire_offset_from_top: the distance of the first wires from the top of the sample
        :param wire_interdistance_thickness: the distance between wires in the thickness direction
            (i.e. between different lines)
        :param wire_interdistance_width: the distance between wires in the width direction
            (i.e. between wires in the same line)
        """
        wire_instance_name = "Wire"
        root_assembly = self.__model.rootAssembly
        first_wire_instance = root_assembly.Instance(dependent=ON, name=wire_instance_name, part=self.part)
        self.instance = first_wire_instance

        if self.test_type.value["Orientation"] == "Horizontal":
            root_assembly.rotate(angle=-90.0, axisDirection=(0.0, 1.0, 0.0), axisPoint=(0.0, 0.0, 0.0),
                instanceList=(wire_instance_name, ))
        elif self.test_type.value["Orientation"] == "Vertical":
            root_assembly.rotate(angle=90.0, axisDirection=(0.0, 0.0, 1.0), axisPoint=(0.0, 0.0, 0.0),
                instanceList=(wire_instance_name, ))
            # Implicit second rotation of 180 around the global y axis (but turning a 1D object doesn't do anything)

        def rotate_coords(coords: tuple[Quantity, Quantity, Quantity] | tuple[float, float, float]) \
                -> tuple[Quantity, Quantity, Quantity] | tuple[float, float, float]:
            """Rotate coordinates from the VERTICAL to HORIZONTAL following the rotation defined in the
            ``add_to_assembly`` method."""
            x, y, z = coords
            if self.test_type.value["Orientation"] == "Horizontal":
                return z, y, -x
            elif self.test_type.value["Orientation"] == "Vertical":
                return z, x, -y  # Result of 2 rotations: 90 around (0,0,1) then 180 around (0,1,0)
            else:
                raise ValueError(f"Unsupported orientation {self.test_type.value['Orientation']}.")

        root_assembly.translate(instanceList=(wire_instance_name, ),
            vector=rotate_coords((0.0, wire_offset_from_top.m, wire_offset_from_side.m)))
        edges = first_wire_instance.edges

        if self.test_type.value["Wire"] != "Single":
            if n_lines > 1 and n_wires % n_lines != 0:
                raise ValueError("n_wires must be divisible by the number of lines to allow proper distribution.")
            if self.__symmetry_lateral and (n_wires // n_lines) % 2 != 0:
                raise ValueError("n_wires// n_lines must be an even number to ensure placement with lateral symmetry.")
            wires_per_line = n_wires // n_lines
            if self.__symmetry_lateral:
                wires_per_line //= 2

            root_assembly.LinearInstancePattern(instanceList=(wire_instance_name, ),
                direction1=rotate_coords((0.0, 1.0, 0.0)), direction2=rotate_coords((0.0, 0.0, 1.0)),
                number1=n_lines, number2=wires_per_line,
                spacing1=wire_interdistance_thickness.m, spacing2=wire_interdistance_width.m)

            for i in range(1, n_lines + 1):
                for j in range(2 if i == 1 else 1, wires_per_line+1):
                    edges += root_assembly.instances[f"{wire_instance_name}-lin-{i}-{j}"].edges

            self.info["Configuration"] = f"{n_lines} lines with {wires_per_line} wires per line"

        root_assembly.regenerate()
        self.set_all_wires = root_assembly.Set(edges=edges, name="WiresSet")

        self.__added_to_assembly = True
        self.info["Number of wires"] = n_wires

    def embed_wires_in_host(self, set_host: object) -> None:
        """
        Create an embedded region to embed the wires in the host FRP sample.
         The wires will be embedded in the host.

         :param set_host: the set of the host region in which the wires will be embedded
            (e.g. the whole sample or just a part of it)
         """
        if not self.__added_to_assembly:
            raise RuntimeError("Wires must be added to the assembly before being embedded in the host.")

        self.__model.EmbeddedRegion(embeddedRegion=self.set_all_wires,
            hostRegion=set_host, name="FRP_SMA_bond", weightFactorTolerance=1e-06, fractionalTolerance=0.05)

    def prestrain(self, stepname: str, prestrain_value: Quantity, create_output_request: bool = False):
        """
        Apply prestrain to the wires by creating a temperature field that will induce the desired prestrain based on
        the coefficient of thermal expansion of the wire material.

        :param stepname: the name of the step (already created in the model) in which the prestrain will be applied.
        :param prestrain_value: the desired prestrain value (e.g. 0.05*ureg.dimensionless or 5*ureg.percent for 5%
            prestrain).
        :param create_output_request: whether to create a history output request to monitor the stress, strain and
            temperature in the wires during the analysis. Default is False, because this inflates the output database
            size significantly, but should be used at least once on a model to check the prestrain is applied correctly.
        :return:
        """
        if not self.__added_to_assembly:
            raise RuntimeError("Wires must be added to the assembly before applying prestrain.")

        if self.__model.wire_material.alpha is None:
            raise RuntimeWarning("Coefficient of thermal expansion (alpha) is not defined for the wire material."
                                 "Prestrain will have no effect.")
        prestrain_value.ito(ureg.dimensionless)  # Ensure prestrain value is dimensionless
        deltaT = (-prestrain_value / self.__model.wire_material.alpha).m
        tempfield = self.__model.Temperature(createStepName='Initial', name="PrestrainTemperature",
            region=self.set_all_wires, magnitudes=(0.0, ))
        tempfield.setValuesInStep(magnitudes=(deltaT, ), stepName=stepname)

        if create_output_request:
            self.__model.HistoryOutputRequest(createStepName=stepname, name="PrestrainWires", rebar=EXCLUDE,
                region=self.set_all_wires, sectionPoints=DEFAULT, variables=("S11", "LE11", "THE11", "TEMP"))

        self.info["Prestrain"] = prestrain_value.dict

    def info(self) -> dict[str, dict | str | bool]:
        return self.info


class ModelWithWires(Model):
    def __init__(self, name: str, directory: Path, symmetry: str | None, **kwargs):
        """
        Create a model that can include Wires. (This class is created to avoid making the main Model class less
        generic but also at the same time avoid having to duplicate a lot of code related to the wires in the
        Test classes like ThreePointBendingTestWithWires, TensionTestWithWires and CompressionTestWithWires).

        :param name: the name of the model
        :param directory: the directory in which the model will be saved (as a .cae file)
        :param symmetry: the symmetry of the model, which can be "lateral", "longitudinal",
            "lateral+longitudinal" or None (no symmetry).
        :param kwargs: extra keyword arguments to pass to the Model constructor (e.g. model_type, etc.)
        """
        model_type = kwargs.pop("model_type", "Standard")
        super().__init__(name=name, directory=directory, model_type=model_type, **kwargs)

        self.set_loading = None

        self.job_name = None

        self.wires = None
        self.wire_diameter = None
        self.wire_material = None

        self.test_type = None
        self.sample_width = None
        self.sample_length = None
        self.sample_thickness = None
        self.frp_sample = None
        self.__symmetry = symmetry

    def add_wires(self, wire_material: Callable[[Model], SMAMaterial | StainlessSteelMaterial],
            wire_percentage: Quantity, wire_prestrain: Quantity | None = None,
            wire_configuration: str = "offcentred", wire_offset_from_top: Quantity = 0.25 * ureg.mm,
            wire_mesh_size: Quantity = 0.325 * ureg.mm, **kwargs) -> Quantity:
        """
        Add wires to the model by creating a new part for the wires, meshing it with beam elements, and embedding it
        in the host FRP sample using an embedded region. The wires will be distributed in the width and thickness
        following the specified parameters.

        :param wire_material: the material to use for the wires.
        :param wire_percentage: the desired percentage of wires in the sample cross-section (e.g.
            0.01*ureg.dimensionless or 1*ureg.percent for 1%).
        :param wire_prestrain: the desired prestrain to apply to the wires. If None, no prestrain will be applied.
            If specified, a step named "WirePrestrain" will be created in which the prestrain is applied
        :param wire_configuration: the configuration of the wires in the cross-section. If "offcentred", wires will
            be placed in two lines close to the outer faces of the sample (i.e. with an offset from the top and
            bottom of the sample defined by ``wire_offset_from_top``). If "centred", all wires will be placed in
            the middle of the sample thickness (i.e. with an offset from the top and bottom of the sample equal to
            half of the sample thickness). Default is "offcentred".
        :param wire_offset_from_top: the distance of the first wires from the top of the sample, if the "offcentred"
        configuration is chosen.
        :param wire_mesh_size: the desired mesh size for the wires.
        :param kwargs:
        :return:
        """
        self.wire_diameter = kwargs.get("wire_diameter", 0.125 * ureg.mm)
        self.wire_material = wire_material(self)
        n_wires, actual_percentage = wire_number_from_percentage(wire_percentage, self.wire_diameter,
            self.sample_width, self.sample_thickness, ensure_div_4=True)

        if wire_configuration == "offcentred":
            wires_per_line = n_wires // 2
            wire_interdistance_thickness = self.sample_thickness - 2 * wire_offset_from_top
            n_lines = 2
        elif wire_configuration == "centred":
            wires_per_line = n_wires  # All wires are in the middle
            wire_interdistance_thickness = 1 * ureg.mm  # Doesn't matter, but set to non-zero for easy error detection
            wire_offset_from_top = self.sample_thickness / 2
            n_lines = 1
        else:
            raise ValueError("Invalid wire configuration. Choose either 'offcentred' or 'centred'.")

        wire_interdistance_horizontal = self.sample_width / wires_per_line
        wire_offset_from_side = wire_interdistance_horizontal / 2

        wires = WiresSample(model=self, test_type=self.test_type, symmetry=self.__symmetry)
        wires.create_geometry(length=self.sample_length)
        wires.assign_section(diameter=self.wire_diameter, material=self.wire_material)
        wires.mesh_part(mesh_size=wire_mesh_size)
        wires.add_to_assembly(n_wires=n_wires, n_lines=n_lines, wire_offset_from_side=wire_offset_from_side,
            wire_offset_from_top=wire_offset_from_top, wire_interdistance_width=wire_interdistance_horizontal,
            wire_interdistance_thickness=wire_interdistance_thickness)

        if self.frp_sample is not None:
            wires.embed_wires_in_host(set_host=self.frp_sample.set_frp_sample)

        self.wires = wires

        if wire_prestrain is not None and wire_prestrain != 0*ureg.percent:
            wires.prestrain(stepname="WirePrestrain", prestrain_value=wire_prestrain, create_output_request=kwargs.get("create_output_request", False))

        return actual_percentage

    def save(self):
        """Create the job and save the model to a .cae file."""
        self.job_name = self.create_job()
        save_mdb_to_cae_file(self.directory / f"{self.name}.cae")

    def _history_output(self, stepname: str):
        """
        Create history output request to monitor the vertical reaction force and displacement at the loading tip
        during the bending step. Depending on the loading type, the output request will be created either on the
        set of the loading region (if direct BC) or on the reference point of the analytical surface representing
        the loading tip (if analytical tip).

        :param stepname: the step during which to create the output request
        """
        del self.historyOutputRequests["H-Output-1"]  # Remove default
        self.HistoryOutputRequest(createStepName=stepname, name="ForceDisplacement",
            region=self.set_loading, variables=("U2", "RF2"), frequency=1)

    def _field_output(self, stepname: str, damage_type: str | None = None, output_all_integration_points: bool = False):
        """Create field output request to monitor the stress, strain and other relevant variables.

        :param stepname: the step during which to create the output request
        :param damage_type: the type of damage model used in the analysis, which determines the damage variables
            to include in the output request. Can be "Hashin", "PlyFabric".
        :param output_all_integration_points: whether to output the field variables at all integration points in the
            FRP sample (instead of just the default output points, which are the top and bottom faces of the sample).
        """
        del self.fieldOutputRequests["F-Output-1"]
        self.FieldOutputRequest(createStepName=stepname, variables=("S", "PE", "LE", "U", "RF"),
            name="DefaultFieldOutput")
        field_output_vars = ("CF", "CSTRESS")
        if damage_type == "Hashin":
            field_output_vars += ("DAMAGEFT", "DAMAGEFC", "DAMAGEMT", "DAMAGEMC", "DAMAGESHR")
        elif damage_type == "PlyFabric":
            field_output_vars += ("DAMAGEF1C", "DAMAGEF1T", "DAMAGEF2C", "DAMAGEF2T")
        self.FieldOutputRequest(createStepName=stepname, region=self.frp_sample.set_frp_sample,
            variables=field_output_vars, name="FRPFieldOutput")

        if output_all_integration_points:
            self.FieldOutputRequest(name="FRPAllIntegrationPointOutput", createStepName=stepname,
            layupLocationMethod=ALL_LOCATIONS, layupNames=(self.frp_sample.nam + "." + self.frp_sample._layup_name,), rebar=EXCLUDE,
                variables=("S", "PE", "LE", "DAMAGEFT", "DAMAGEFC", "DAMAGEMT", "DAMAGEMC", "DAMAGESHR"))


class WireTensileTest(ModelWithWires):
    def __init__(self, directory: Path,
            wire_material: Callable[[Model], SMAMaterial | StainlessSteelMaterial],
            wire_prestrain: Quantity | None = None,
            name: str = "Wire tensile test",
            mesh_size: Quantity = 5 * ureg.mm,
            max_displacement: Quantity = 20 * ureg.mm,
            loading_amplitude: tuple[float, Quantity] | None = None,
            **kwargs):
        """
        Create an Abaqus model for a tensile test on a one single wire. Boundary conditions are applied directly on
        the outer nodes of the wire, with one end
        fixed and the other end loaded with a vertical displacement. The mesh is created with truss elements.
        """

        super().__init__(name=name, directory=directory, symmetry=None, **kwargs)

        self.test_type = TestType.WIRE_TENSION
        self.__symmetry = None

        self.sample_length = kwargs.get("sample_length", 100 * ureg.mm)
        self.wire_diameter = kwargs.get("wire_diameter", 0.125 * ureg.mm)
        self.wire_material = wire_material(self)

        wire = WiresSample(model=self, test_type=self.test_type, symmetry=self.__symmetry)
        wire.create_geometry(length=self.sample_length)
        wire.assign_section(diameter=self.wire_diameter, material=self.wire_material)
        wire.mesh_part(mesh_size=mesh_size)
        wire.add_to_assembly(n_wires=1, n_lines=1, wire_offset_from_side=0*ureg.mm,
            wire_offset_from_top=0*ureg.mm, wire_interdistance_width=0*ureg.mm,
            wire_interdistance_thickness=0*ureg.mm)

        self.wire = wire
        self.loading_set_name = "WireLoadingSet"

        self.set_loading = self.rootAssembly.Set(vertices=find_geometry_by_coords(self.wire.instance,
            (0*ureg.mm, self.sample_length, 0*ureg.mm), GeometryType.VERTEX), name="WireLoadingSet")
        self.set_support = self.rootAssembly.Set(vertices=find_geometry_by_coords(self.wire.instance,
            (0*ureg.mm, 0*ureg.mm, 0*ureg.mm), GeometryType.VERTEX), name="WireSupportSet")
        self.set_wire = self.rootAssembly.Set(edges=self.wire.instance.edges, name="WireSet")

        self.__bc_initial()

        if wire_prestrain is not None and wire_prestrain != 0*ureg.percent:
            raise NotImplementedError("Prestrain is currently not implemented for the wire tensile test")
            # Needs to define something to hold back the wire while prestraining
            self.create_step(name="WirePrestrain")
            wire.prestrain(stepname="WirePrestrain", prestrain_value=wire_prestrain, create_output_request=False)

        if loading_amplitude is not None:
            amplitude = self.TabularAmplitude(data=[(a[0], a[1].m) for a in loading_amplitude],
                name="QuasiStaticAmplitude", smooth=SOLVER_DEFAULT, timeSpan=STEP)
            self.__bc_tension(stepname="TensionTest", max_disp=1 * ureg.mm, amplitude=amplitude.name, step_time=loading_amplitude[-1][0])
        else:
            self.__bc_tension(stepname="TensionTest", max_disp=max_displacement)
        self.__history_output(stepname="TensionTest")

    def __bc_initial(self):
        """Apply the initial boundary conditions for the wire tensile test: one end of the wire is fixed, and the other
        end is loaded with a vertical displacement."""
        self.EncastreBC(createStepName="Initial", name="FixedEnd", region=self.set_support)
        self.__BC_loading_tip = self.DisplacementBC(createStepName="Initial", name="LoadedEnd",
            region=self.set_loading, u1=UNSET, u2=SET, u3=UNSET)
        self.DisplacementBC(createStepName="Initial", name="ConstrainZeroStifnessDirections",
            region=self.set_wire, u1=SET, u3=SET)

    def __bc_tension(self, stepname: str, max_disp: Quantity, amplitude: str = "", step_time: float = 1):
        """Apply the boundary conditions for the tension step: a vertical displacement is applied on one end of the
        wire, while the other end remains fixed."""
        self.create_step(name=stepname, minimum_number_of_increments=75, step_time=step_time)
        self.__BC_loading_tip.setValuesInStep(stepName=stepname, u2=max_disp.m, amplitude=amplitude)

    def __history_output(self, stepname: str):
        """Create history output request to monitor the vertical reaction force and displacement at the loaded end of
        the wire during the tension step."""
        del self.historyOutputRequests["H-Output-1"]  # Remove default
        self.HistoryOutputRequest(createStepName=stepname, name="ForceDisplacement",
            region=self.set_loading, variables=("U2", "RF2"))

    def extract_results(self, results_dir: Path, false_run: bool = False):
        """Extract the reaction force and displacement at the loaded end of the wire from the output database after
        running the analysis."""
        if false_run:
            return

        rp_set_name = self.loading_set_name

        sample_data = {
            "Test type": {"Type": "wire tensile test"},
            "Material": self.wire_material.info,
            "Sample type": {"Wire": self.wire.info},
        }
        extract_history_output(self.directory / f"{self.job.name}.odb", rp_set_name.upper(), results_dir, sample_data,
            extract_tensile_results, "TensionTest", symmetry_factor=1, diameter=self.wire_diameter,
            gauge_length=self.sample_length, wire_only_test=True)


class ThreePointBendingTest(ModelWithWires):
    def __init__(self, directory: Path,
            frp_material: Callable[[Model], FRPMaterial],
            name: str = "3-point bending test",
            coarse_mesh_size: Quantity = 1.25 * ureg.mm, fine_mesh_size: Quantity | None = None,
            symmetry: str = "lateral+longitudinal",
            support_type: str = "Direct BC",
            loading_type: str = "Analytical tip",
            loading_amplitude: list[tuple[float, Quantity], ...] | None = None,
            include_wire_prestrain_step: bool = False,
            **kwargs):
        """
        Create an Abaqus model for a 3-point bending test on an FRP sample. The sample geometry is a simple
        rectangle. Boundary conditions can be applied directly or by creating analytical rigid surface tips at the
        supports and loading points, with contact interactions. The mesh is created with continuum shell elements
        (SC8R) and a structured mesh, with the possibility to have different mesh sizes in different zones of the
        sample.

        :param frp_material: the material fabric to create the FRP material for the sample
        :param name: the name of the model
        :param coarse_mesh_size: desired mesh size at the supports and further away from the loading point
        :param fine_mesh_size: desired mesh size at the centre of the sample (where loading is applied). If
            None, no finer mesh will be applied at the centre (the partitions will still be created though).
        :param symmetry: the symmetry of the model. Can be "lateral", "longitudinal", "lateral+longitudinal" or
            None (no symmetry). *Note: currently longitudinal is mandatory.*
        :param support_type: the type of supports to create. Can be "Direct BC" for direct boundary conditions on the
            faces/edges at the support locations, or "Analytical tip" to create analytical rigid surface tips
        :param loading_type: the type of loading to create. Can be "Face/edge load" to apply the load directly on the
            faces/edges at the loading location, or "Analytical tip" to create an analytical rigid surface tip
        :param loading_amplitude: if using an analytical tip for loading, the amplitude of the loading to apply on
            the tip as a table with (time, value) tuples (e.g. [(0, 0*ureg.N), (1, 10*ureg.N)] for a load that goes
            from 0 to 10N in 1 step).
        """

        super().__init__(name=name, directory=directory, symmetry=symmetry, **kwargs)

        self.test_type = TestType.THREE_POINT_BENDING
        self.__symmetry = symmetry
        if "longitudinal" not in symmetry:
            raise NotImplementedError("3 Point Bending is currently not yet implemented without longitudinal symmetry")

        self.__support_type = support_type
        self.__loading_type = loading_type

        self.frp_material = frp_material(self)

        self.sample_length = kwargs.get("full_sample_length", 115 * ureg.mm)  # Normative value is 100±10mm, but use Yvan's dimensions
        self.sample_width = kwargs.get("full_sample_width", 15 * ureg.mm)  # Normative value
        self.sample_thickness = kwargs.get("full_sample_thickness", 2 * ureg.mm)  # Normative value
        self.inter_support_length = kwargs.get("inter_support_length", 100 * ureg.mm)  # Normative value

        model_width = self.sample_width if "lateral" not in symmetry else self.sample_width / 2
        model_length = self.sample_length if "longitudinal" not in symmetry else self.sample_length / 2

        if fine_mesh_size is not None:
            fine_mesh_length = kwargs.get("fine_mesh_length", min(5 * ureg.mm, fine_mesh_size * 10))  # Length of the zone where finer mesh is desired (around the loading point)
            transition_mesh_length = kwargs.get("transition_mesh_length", min(15 * ureg.mm, fine_mesh_size * 20))  # Length of the transition zone between finer and coarser meshes
        else:
            fine_mesh_size = coarse_mesh_size
            fine_mesh_length = 1 * fine_mesh_size  # Ensure partitions don't alter seeding
            transition_mesh_length = 1 * fine_mesh_size  # Ensure partitions don't alter seeding

        if (round((self.sample_width/2) / fine_mesh_size, 0) - round((self.sample_width/2) / coarse_mesh_size, 0)) % 2 != 0:
            raise ValueError("To ensure the transition between finer mesh and bigger mesh, please ensure the difference in number of elements between the finer and coarser meshes is an even number (so that the transition can be symmetric on both sides of the finer mesh zone).")

        # Zone outside of the supports, zone inside supports but away from fine mesh, zone inside supports with transition mesh, zone close to loading tip with fine mesh
        partitions = [(self.sample_length - self.inter_support_length)/2, self.sample_length/2-fine_mesh_length-transition_mesh_length, self.sample_length/2-fine_mesh_length]
        frp_sample = FRPSample(model=self, test_type=self.test_type, symmetry=symmetry)
        frp_sample.create_geometry(width=self.sample_width, length=self.sample_length, thickness=self.sample_thickness,
            partitions=partitions, partition_support_line=partitions[0])
        frp_sample.assign_composite_layup_section(material=self.frp_material, **kwargs)
        frp_sample.mesh_part(coarse_mesh_size=coarse_mesh_size, fine_mesh_size=fine_mesh_size,
            fine_mesh_cells=[self.sample_length/2-fine_mesh_length/2], **kwargs)
        frp_sample.add_to_assembly()

        self.frp_sample = frp_sample

        self.set_supports = self.frp_sample.set_supports
        self.set_loading = self.frp_sample.set_loading
        self.loading_set_name = "LoadingSet"

        # Create contact interactions (in case of analytical tips)
        if self.__loading_type == "Analytical tip" or self.__support_type == "Analytical tip":
            self.interaction_property = self.ContactProperty("RigidBodyContact")
            self.interaction_property.NormalBehavior(allowSeparation=ON, constraintEnforcementMethod=DEFAULT,
                pressureOverclosure=HARD)
            self.interaction_property.TangentialBehavior(formulation=FRICTIONLESS)

        if self.__loading_type == "Analytical tip":
            loading_tip_diameter = kwargs.pop("loading_tip_diameter",
                10*ureg.mm if self.sample_thickness >= 3*ureg.mm else 4*ureg.mm)
            self.set_loading = self.__create_bench_tip(diameter=loading_tip_diameter, length=self.sample_width*1.5,
                name="LoadingTip", transl=(model_width.m*0.5, 2, model_length.m), rotation="down")
            self.loading_set_name = "LoadingTip"

        if self.__support_type == "Analytical tip":
            if "lateral" not in self.__symmetry:
                raise NotImplementedError("Analytical tips for supports are currently only implemented for models with "
                                          "lateral symmetry (i.e. half width models).")
            support_tip_diameter = kwargs.pop("support_tip_diameter", 10*ureg.mm)
            support_edge = self.frp_sample.set_supports.edges[0].pointOn[0]

            self.set_supports = self.__create_bench_tip(diameter=support_tip_diameter, length=self.sample_width*1.5,
                name="SupportTip", transl=(model_width.m*0.5, support_edge[1], support_edge[2]), rotation="up")

        self.__bc_initial(stepname="Initial")

        if include_wire_prestrain_step:
            self.create_step(name="WirePrestrain")  # Ensure steps created in right order

        if loading_amplitude is not None:
            amplitude = self.TabularAmplitude(data=[(a[0], a[1].m) for a in loading_amplitude],
                name="QuasiStaticAmplitude", smooth=SOLVER_DEFAULT, timeSpan=STEP)
            self.__bc_bending_test(stepname="BendingTest", max_disp=1 * ureg.mm, amplitude=amplitude.name, step_time=loading_amplitude[-1][0])
        else:
            self.__bc_bending_test(stepname="BendingTest")
        self._history_output(stepname="BendingTest")

        self._field_output(stepname="BendingTest", damage_type=self.frp_material._damage_initiation_criterion,
            output_all_integration_points=kwargs.get("output_all_integration_points", False))

    def __create_bench_tip(self, diameter: Quantity, length: Quantity, name: str, transl, rotation: str = "down"):
        """
        Create an analytical rigid surface in the shape of a cylindrical tip (with a rounded end) to represent the
        loading or support tip in the bending test, and create a surface-to-surface contact interaction between the
        tip and the sample.

        :param diameter: the diameter of the tip
        :param length: the length of the tip (set slightly larger than sample width to ensure full contact with
            the sample face even after small sample deformations)
        :param name: the name
        :param transl: a translation to apply to the tip after creation, to position it correctly very close to the
            sample face. Should be a tuple of 3 values (x, y, z) in the model coordinate system. Initially the
            tip is created with the centre of its tip at (0, 0, 0).
        :param rotation: "down" to have the tip facing downwards (i.e. the loading tip), "up" to have the tip facing
            upwards (i.e. the support tips)
        :return: the set containing the reference point of the tip, which can be used later to apply boundary
            conditions
        """
        if "lateral" in self.__symmetry:
            length /= 2  # Because sample will be shorter as well
        ##
        # Create part
        ##
        sketch = self.ConstrainedSketch(name='__profile__', sheetSize=200.0)

        # Avoid arc of 180°, so make it only 170°, then correct the radius
        y = diameter/2+diameter/2*np.sin(np.deg2rad(-5))
        x = diameter/2*np.cos(np.deg2rad(-5))
        sketch.ArcByCenterEnds(center=(0.0, (diameter/2).m), direction=COUNTERCLOCKWISE, point1=(-x.m, y.m),
            point2=(x.m, y.m))
        sketch.RadialDimension(curve=sketch.geometry.findAt((x.m, y.m), ), radius=(diameter/2).m,
            textPoint=(0.0, (diameter/2).m))

        part = self.Part(dimensionality=THREE_D, name=name, type=ANALYTIC_RIGID_SURFACE)
        part.AnalyticRigidSurfExtrude(depth=length.m, sketch=sketch)
        del sketch

        # Ensure partition to align later on with the partitions on the sample face
        part.PartitionFaceByShortestPath(faces=part.faces.findAt(((0, 0, 0), )),
            point1=part.InterestingPoint(part.edges.findAt((0, 0, (length/2).m), ), MIDDLE),
            point2=part.InterestingPoint(part.edges.findAt((0, 0, -(length/2).m), ), MIDDLE))

        part.ReferencePoint(point=part.InterestingPoint(part.edges.findAt((0, 0, (length/2).m), ), CENTER))

        ##
        # Create instance
        ##
        root_assembly = self.rootAssembly
        instance_tip = root_assembly.Instance(dependent=ON, name=name, part=part)

        root_assembly.rotate(angle=90.0, axisDirection=(0.0, 1.0, 0.0), axisPoint=(0.0, 0.0, 0.0),
            instanceList=(instance_tip.name, ))
        if rotation == "up":
            root_assembly.rotate(angle=180, axisDirection=(1.0, 0.0, 0.0), axisPoint=(0.0, 0.0, 0.0),
                instanceList=(instance_tip.name, ))  # To have the round face of the tip facing the sample

        root_assembly.translate(instanceList=(instance_tip.name, ), vector=transl)
        root_assembly.regenerate()

        tip_surface = root_assembly.Surface(name="SupportTipSurface", side2Faces=instance_tip.faces)
        frp_surface = root_assembly.Surface(name="FRPSupportSurface", side1Faces=self.frp_sample.instance.faces)

        self.SurfaceToSurfaceContactStd(contactTracking=ONE_CONFIG, createStepName="Initial",
            interactionProperty=self.interaction_property.name, main=tip_surface,
            name=f"{name}Contact", secondary=frp_surface)

        set_tip = root_assembly.Set(name=name, referencePoints=(instance_tip.referencePoints[3], ))
        
        return set_tip

    def __bc_initial(self, stepname) -> None:
        """
        Apply boundary conditions for the initial step. Depending on the specified support and loading types,
        either directly to a line on the sample, or to a reference point associated with an analytical surface
        representing the loading tip.

        *Note: for continuum shell elements, rotation DOFs are not active, so they are left as UNSET*

        :param stepname: step during which to apply these boundary conditions
        """
        if self.__support_type == "Direct BC":
            # All DOFs except vertical translation are free
            BC_supports = self.DisplacementBC(createStepName=stepname,
                name="BCSupports", region=self.set_supports,
                u1=UNSET, u2=SET, u3=UNSET, ur1=UNSET, ur2=UNSET, ur3=UNSET)
        else:
            # Rigid body motion of the supports is prevented by coupling constraint to reference points
            BC_supports = self.DisplacementBC(createStepName=stepname,
                name="BCSupports", region=self.set_supports,
                u1=SET, u2=SET, u3=SET, ur1=SET, ur2=SET, ur3=SET)

        if self.__loading_type == "Direct BC":
            # All DOFs except vertical translation are free
            BC_loading_tip = self.DisplacementBC(createStepName=stepname,
                name="BCLoadingTip", region=self.set_loading,
                u1=UNSET, u2=SET, u3=UNSET, ur1=UNSET, ur2=UNSET, ur3=UNSET)
        else:
            # Rigid body motion of the loading tip is prevented by coupling constraint to reference point
            BC_loading_tip = self.DisplacementBC(createStepName=stepname,
                name="BCLoadingTip", region=self.set_loading,
                u1=SET, u2=SET, u3=SET, ur1=SET, ur2=SET, ur3=SET)

        self.__BC_loading_tip = BC_loading_tip
        self.__BC_supports = BC_supports

    def __bc_bending_test(self, stepname, max_disp: Quantity = -20 * ureg.mm, amplitude="LinearIncrease",
        step_time: float = 1) -> None:
        """
        Apply the vertical displacement to the loading tip, either directly if it's a direct BC, or to the reference
        point of the analytical surface representing the loading tip.

        :param stepname: the step during which to apply the displacement
        :param max_disp: the maximum displacement to apply at the loading tip (e.g. -20*ureg.mm for 20 mm
            downward displacement)
        :param amplitude: the amplitude to use for the displacement. Default is "LinearIncrease", which means the
            displacement will increase linearly from 0 to the maximum value during the step. If using another
            amplitude, make sure to use max_disp=1*ureg.mm, otherwise results might be unexpected (as Abaqus
            will multiply the amplitude value by the max_disp value to get the actual displacement to apply).
        :param step_time: the total time of the step (only relevant if using an amplitude that is not "LinearIncrease",
            as it will be defined by the time span of the amplitude)
        """
        self.create_step(name=stepname, minimum_number_of_increments=100, step_time=step_time)
        self.__BC_loading_tip.setValuesInStep(stepName=stepname, u2=max_disp.m, amplitude=amplitude)

    def extract_results(self, results_dir: Path, false_run: bool = False):
        """
        Extract the results from the output database and save them in a .csv file in the specified directory. The
        results include a Force-Displacement time series extracted at the loading point. The metadata is saved
        to a .json file in the same directory, and includes the test type, the material and sample information,
        and information on the columns of the results .csv file.

        :param results_dir: the directory in which to save the results .csv file and metadata .json file
        :param false_run: if True, skip the actual extraction and don't modify any existing results (useful for
            rerunning only part of a workflow).
        """
        if false_run:
            return

        rp_set_name = self.loading_set_name
        sample_data = {
            "Test type": {"Type": "3 point bending"},
            "Material": [self.frp_material.info, self.wire_material.info if self.wire_material else None],
            "Sample type": {"FRP Sample": self.frp_sample.info, "Wires": self.wires.info if self.wires else None},
        }
        symmetry_factor = 4 if "lateral" in self.__symmetry and "longitudinal" in self.__symmetry else 2 if "lateral" in self.__symmetry or "longitudinal" in self.__symmetry else 1
        extract_history_output(self.directory / f"{self.job.name}.odb", rp_set_name.upper(), results_dir, sample_data,
            extract_midpoint_force_displacement, "BendingTest", symmetry_factor=symmetry_factor)

    def extract_prestrain_results(self, results_dir: Path, false_run: bool = False):
        if false_run:
            return

        rp_set_name = "WiresSet"
        sample_data = {
            "Test type": {"Type": "3 point bending"},
            "Material": [self.frp_material.info, self.wire_material.info if self.wire_material else None],
            "Sample type": {"FRP Sample": self.frp_sample.info, "Wires": self.wires.info if self.wires else None},
        }
        extract_history_output(self.directory / f"{self.job.name}.odb", rp_set_name.upper(), results_dir, sample_data,
            extract_prestrain_results, "WirePrestrain", allow_multiple_instances=True, strict=False)


class TensionTest(ModelWithWires):
    def __init__(self, directory: Path,
            frp_material: Callable[[Model], FRPMaterial],
            name: str = "Tension test",
            mesh_size: Quantity = 0.75*ureg.mm,
            max_displacement: Quantity = 1.6 * ureg.mm,
            loading_amplitude: list[tuple[float, Quantity]] | None = None,
            test_type: TestType = TestType.TENSION,
            **kwargs):
        """
        Create a simple tension test model with the specified parameters. The tension test consists of a
        rectangular FRP sample (possibly with wires, call `add_wires` method to add them) with the lower end fixed
        and the upper end subjected to a vertical displacement. In this model the displacement is modeled as
        applied directly and uniformly to the top face of the sample. The fixed end can translate in the width
        direction but is blocked by a symmetry plane.


        :param directory: the directory in which to save the model and results
        :param frp_material: the material fabric to create the FRP material for the sample
        :param name: the name of the model
        :param mesh_size: the desired mesh size for the sample
        :param max_displacement: the maximum displacement to apply at the loading end
            (e.g. 1.6*ureg.mm for 1.6 mm upward displacement).
        :param loading_amplitude: the amplitude of the loading to apply, as a table with (time, value) tuples
            (e.g. [(0, 0*ureg.N), (1, 10*ureg.N)]
        :param test_type: only to be used by the CompressionTest class: change to TestType.COMPRESSION
        :param kwargs: additional parameters to create the model, such as the sample dimensions
            (full_sample_length, full_sample_width, ...)
        """

        self.__symmetry = "lateral"
        super().__init__(name=name, directory=directory, symmetry=self.__symmetry, **kwargs)

        self.test_type = test_type
        self.frp_material = frp_material(self)

        self.sample_length = kwargs.get("full_sample_length", 50 * ureg.mm)  # Only reference length
        self.sample_width = kwargs.get("full_sample_width", 10 * ureg.mm)
        self.sample_thickness = kwargs.get("full_sample_thickness", 2 * ureg.mm)

        frp_sample = FRPSample(model=self, test_type=self.test_type, symmetry=self.__symmetry)
        frp_sample.create_geometry(width=self.sample_width, length=self.sample_length, thickness=self.sample_thickness,
            partitions=None)
        frp_sample.assign_composite_layup_section(material=self.frp_material, **kwargs)

        kwargs.setdefault("coarse_mesh_size", mesh_size)
        frp_sample.mesh_part(**kwargs)
        frp_sample.add_to_assembly()

        self.frp_sample = frp_sample

        self.set_supports = self.frp_sample.set_supports
        self.set_loading = self.frp_sample.set_loading
        self.loading_set_name = "LoadingSet"

        # Create contact interactions (in case of analytical tips)
        self.__loading_bc = None
        self.__bc_initial(stepname="Initial")

        if loading_amplitude is not None:
            amplitude = self.TabularAmplitude(data=[(a[0], a[1].m) for a in loading_amplitude],
                name="QuasiStaticAmplitude", smooth=SOLVER_DEFAULT, timeSpan=STEP)
            self.__bc_tensile_test(stepname="Loading", max_disp=max_displacement, amplitude=amplitude.name)
        else:
            self.__bc_tensile_test(stepname="Loading", max_disp=max_displacement)
        self._history_output(stepname="Loading")

        self._field_output(stepname="Loading", damage_type=self.frp_material._damage_initiation_criterion)

    def __bc_initial(self, stepname) -> None:
        """
        Apply the initial boundary conditions for the tension test: the bottom face of the sample is fixed except in
        the width direction (but that is already blocked by a symmetry constrained created in the `FRPSample` class).
        The top face of the sample is free in all directions except vertically and in the thickness direction,
        and will be subjected to a vertical displacement in the loading step.

        :param stepname: the step during which to apply these initial boundary conditions
        """
        self.DisplacementBC(createStepName=stepname, name="EncastreBottom", region=self.set_supports,
            u1=UNSET, u2=SET, u3=SET, ur1=UNSET, ur2=UNSET, ur3=UNSET)
        self.__loading_bc = self.DisplacementBC(createStepName=stepname, name="Loading", region=self.set_loading,
            u1=UNSET, u2=SET, u3=SET, ur1=UNSET, ur2=UNSET, ur3=UNSET)

    def __bc_tensile_test(self, stepname, max_disp: Quantity = 1 * ureg.mm, amplitude: str ="LinearIncrease") -> None:
        """
        Apply the vertical displacement to the top face of the sample to simulate the tensile loading.

        :param stepname: the step during which to apply the displacement
        :param max_disp: the maximum displacement to apply at the loading end (e.g. 1*ureg.mm for
            1 mm upward displacement). Note that if an amplitude is used, the actual displacement applied will be the
            product of this max_disp value and the amplitude values!
        :param amplitude: an amplitude to use for the displacement. Default is "LinearIncrease", which means the
            displacement will increase linearly from 0 to the maximum value during the step.
        """
        self.create_step(name=stepname, minimum_number_of_increments=100)
        self.__loading_bc.setValuesInStep(stepName=stepname, u2=max_disp.m, amplitude=amplitude)

    def extract_results(self, results_dir: Path, false_run: bool = False):
        """
        Extract the results from the output database and save them in a .csv file in the specified directory. The
        results include a Force-Displacement time series extracted at the loading face, as well
        as the stress-strain time series. The metadata is saved
        to a .json file in the same directory, and includes the test type, the material and sample information,
        and information on the columns of the results .csv file.

        :param results_dir: the directory in which to save the results .csv file and metadata .json file
        :param false_run: if True, skip the actual extraction and don't modify any existing results (useful for
            rerunning only part of a workflow).
        """
        if false_run:
            return

        rp_set_name = self.loading_set_name
        sample_data = {
            "Test type": {"Type": "Tensile test"},
            "Material": [self.frp_material.info, self.wire_material.info if self.wire_material else None],
            "Sample type": {"FRP Sample": self.frp_sample.info, "Wires": self.wires.info if self.wires else None},
        }

        extract_history_output(self.directory / f"{self.job.name}.odb", rp_set_name.upper(), results_dir, sample_data,
            extract_tensile_results, "Loading", width=self.sample_width, thickness=self.sample_thickness,
            gauge_length=self.sample_length, symmetry_factor=2)


class CompressionTest(TensionTest):
    def __init__(self, directory: Path,
            frp_material: Callable[[Model], FRPMaterial],
            name: str = "Compressive test",
            mesh_size: Quantity = 0.75*ureg.mm,
            loading_amplitude: list[tuple[float, Quantity]] | None = None,
            **kwargs):
        """
        Create a simple compression test model (very similar to the tension test, but with the loading applied in the
        opposite direction).

        *Implementation note: as the boundary conditions etc. for a compression test are very similar to a tension
        test, this class inherits from TensionTest and only modifies the geometry and postprocessing.*

        :param directory:
        :param frp_material:
        :param name:
        :param mesh_size:
        :param loading_amplitude:
        :param kwargs:
        """
        kwargs.setdefault("full_sample_length", 13 * ureg.mm)  # Only reference length
        kwargs.setdefault("full_sample_width", 12 * ureg.mm)
        kwargs.setdefault("full_sample_thickness", 2 * ureg.mm)
        kwargs.setdefault("max_displacement", -13*0.012 * ureg.mm)

        super().__init__(directory=directory, frp_material=frp_material, name=name, mesh_size=mesh_size,
            loading_amplitude=loading_amplitude, test_type=TestType.COMPRESSION,
            **kwargs)

    def extract_results(self, results_dir: Path, false_run: bool = False):
        """
        Extract the results from the output database and save them in a .csv file in the specified directory. The
        results include a Force-Displacement time series extracted at the loading face, as well
        as the stress-strain time series. The metadata is saved
        to a .json file in the same directory, and includes the test type, the material and sample information,
        and information on the columns of the results .csv file.

        :param results_dir: the directory in which to save the results .csv file and metadata .json file
        :param false_run: if True, skip the actual extraction and don't modify any existing results (useful for
            rerunning only part of a workflow).
        """
        if false_run:
            return

        rp_set_name = self.loading_set_name
        sample_data = {
            "Test type": {"Type": "Compressive test"},
            "Material": [self.frp_material.info, self.wire_material.info if self.wire_material else None],
            "Sample type": {"FRP Sample": self.frp_sample.info, "Wires": self.wires.info if self.wires else None},
        }

        extract_history_output(self.directory / f"{self.job.name}.odb", rp_set_name.upper(), results_dir, sample_data,
            extract_tensile_results, "Loading", width=self.sample_width, thickness=self.sample_thickness,
            gauge_length=self.sample_length, symmetry_factor=2)


class DynamicTest(ModelWithWires):
    def __init__(self, directory: Path,
            frp_material: Callable[[Model], FRPMaterial],
            wire_material: Callable[[Model], SMAMaterial | StainlessSteelMaterial] | None = None,
            wire_percentage: Quantity | None = None, wire_prestrain: Quantity | None = None,
            wire_configuration: str = "offcentred",
            name: str = "FrequencyTest", **kwargs):
        super().__init__(name=name, directory=directory, symmetry=None, **kwargs)
        self.test_type = TestType.DYNAMIC

        self.frp_material = frp_material(super())
        self.wire_material = wire_material(super()) if wire_material else None

        self.sample_width = kwargs.get("full_sample_width", 15 * ureg.mm)
        self.sample_thickness = kwargs.get("full_sample_thickness", 2 * ureg.mm)
        self.sample_length = kwargs.get("full_sample_length", 200 * ureg.mm)  # 230 mm - 30 mm clamping length

        frp_sample = FRPSample(self, self.test_type, symmetry=None)
        frp_sample.create_geometry(length=self.sample_length, width=self.sample_width, thickness=self.sample_thickness)
        frp_sample.assign_composite_layup_section(self.frp_material)
        frp_sample.mesh_part(20*ureg.mm)
        frp_sample.add_to_assembly()
        self.frp_sample = frp_sample

        if wire_material is not None and wire_percentage is not None:
            self.create_step(name="WirePrestrain")
            self.add_wires(wire_material=wire_material, wire_percentage=wire_percentage, wire_prestrain=wire_prestrain,
                wire_configuration=wire_configuration, wire_mesh_size=20*ureg.mm)

        self.__create_frequency_test_job()

    def __create_frequency_test_job(self):
        """Create the boundary conditions and job for the frequency analysis."""
        super().create_frequency_step(name="Frequency")
        super().EncastreBC(createStepName="Frequency", name="EncastreBottom", region=self.frp_sample.set_supports)
        super().create_job()

    def extract_results(self, results_dir: Path, false_run: bool = False):
        """Extract the eigenfrequencies and save them to a file. Note: for the mode shapes, view the .odb file!"""
        if false_run:
            return

        sample_data = {
            "Test type": {"Type": "Dynamic test - Frequency analysis"},
            "Material": [self.frp_material.info, self.wire_material.info if self.wire_material else None],
            "Sample type": {"FRP Sample": self.frp_sample.info, "Wires": self.wires.info if self.wires else None},
        }
        extract_eigenfrequencies(self.directory / f"{self.job.name}.odb", "Frequency",
            results_dir, sample_data)

