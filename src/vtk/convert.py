
import vtk

def convert_vtu_to_vtp_vtk(vtu_data):
    """
    Converts a VTU object to VTP, with internal checks to ensure 
    valid input and non-empty output.
    """
    # 1. Check if the input is valid
    if vtu_data is None:
        raise ValueError("Input vtu_data is None.")
        
    print(f"Input object type: {vtu_data.GetClassName()}")
    print(f"Input points: {vtu_data.GetNumberOfPoints()}")
    
    if vtu_data.GetNumberOfPoints() == 0:
        raise ValueError("Input VTU has 0 points. Conversion aborted.")

    # 2. Setup the filter
    surface_filter = vtk.vtkDataSetSurfaceFilter()
    surface_filter.SetInputData(vtu_data)
    surface_filter.Update()
    
    # 3. Retrieve the output
    vtp_output = surface_filter.GetOutput()
    
    # 4. Check if the filter actually produced something
    if vtp_output is None or vtp_output.GetNumberOfPoints() == 0:
        raise RuntimeError("Surface extraction failed: The resulting VTP is empty.")
    
    print(f"Conversion successful. Output points: {vtp_output.GetNumberOfPoints()}")
    
    return vtp_output
