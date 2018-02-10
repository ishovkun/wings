#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'

def retreive_data(pvd_file, output_csv_file, n_points=100):
    paraview.simple._DisableFirstRenderCameraReset()

    # create a new 'PVD Reader'
    # solutionpvd = PVDReader(FileName='/media/wdcdrive/cpp/wings/build/solution/solution.pvd')
    solutionpvd = PVDReader(FileName=pvd_file)

    # get animation scene
    animationScene1 = GetAnimationScene()

    # update animation scene based on data timesteps
    animationScene1.UpdateAnimationUsingDataTimeSteps()

    animationScene1.GoToLast()

    # create a new 'Plot Over Line'
    plotOverLine1 = PlotOverLine(Input=solutionpvd,
        Source='High Resolution Line Source')

    plotOverLine1.Source.Resolution = n_points

    # init the 'High Resolution Line Source' selected for 'Source'
    # plotOverLine1.Source.Point1 = [0.0, -7.62, -3.81]
    # plotOverLine1.Source.Point2 = [155.448, 7.62, 3.81]
    plotOverLine1.Source.Point1 = [0.0, 0.0, -3.81]
    plotOverLine1.Source.Point2 = [155.448, 0.0, 3.81]

    # Properties modified on plotOverLine1
    plotOverLine1.Tolerance = 2.22044604925031e-16

    # get layout
    # viewLayout1 = GetLayout()

    # Create a new 'Line Chart View'
    lineChartView1 = CreateView('XYChartView')
    lineChartView1.ViewSize = [753, 707]

    # place view in the layout
    # viewLayout1.AssignView(2, lineChartView1)

    # show data in view
    plotOverLine1Display = Show(plotOverLine1, lineChartView1)
    # trace defaults for the display properties.
    plotOverLine1Display.CompositeDataSetIndex = [0]
    plotOverLine1Display.UseIndexForXAxis = 0
    plotOverLine1Display.XArrayName = 'arc_length'
    plotOverLine1Display.SeriesVisibility = ['pressure', 'Sw']
    plotOverLine1Display.SeriesLabel = ['arc_length', 'arc_length', 'pressure', 'pressure', 'Sw', 'Sw', 'vtkValidPointMask', 'vtkValidPointMask', 'Points_X', 'Points_X', 'Points_Y', 'Points_Y', 'Points_Z', 'Points_Z', 'Points_Magnitude', 'Points_Magnitude']
    plotOverLine1Display.SeriesColor = ['arc_length', '0', '0', '0', 'pressure', '0.89', '0.1', '0.11', 'Sw', '0.22', '0.49', '0.72', 'vtkValidPointMask', '0.3', '0.69', '0.29', 'Points_X', '0.6', '0.31', '0.64', 'Points_Y', '1', '0.5', '0', 'Points_Z', '0.65', '0.34', '0.16', 'Points_Magnitude', '0', '0', '0']
    plotOverLine1Display.SeriesPlotCorner = ['arc_length', '0', 'pressure', '0', 'Sw', '0', 'vtkValidPointMask', '0', 'Points_X', '0', 'Points_Y', '0', 'Points_Z', '0', 'Points_Magnitude', '0']
    plotOverLine1Display.SeriesLineStyle = ['arc_length', '1', 'pressure', '1', 'Sw', '1', 'vtkValidPointMask', '1', 'Points_X', '1', 'Points_Y', '1', 'Points_Z', '1', 'Points_Magnitude', '1']
    plotOverLine1Display.SeriesLineThickness = ['arc_length', '2', 'pressure', '2', 'Sw', '2', 'vtkValidPointMask', '2', 'Points_X', '2', 'Points_Y', '2', 'Points_Z', '2', 'Points_Magnitude', '2']
    plotOverLine1Display.SeriesMarkerStyle = ['arc_length', '0', 'pressure', '0', 'Sw', '0', 'vtkValidPointMask', '0', 'Points_X', '0', 'Points_Y', '0', 'Points_Z', '0', 'Points_Magnitude', '0']

    # Properties modified on plotOverLine1.Source
    plotOverLine1.Source.Point1 = [0.0, 0.0, -3.81]

    # Properties modified on plotOverLine1.Source
    plotOverLine1.Source.Point2 = [155.448, 0.0, 3.81]

    # set active source
    SetActiveSource(solutionpvd)

    # set active view
    SetActiveView(lineChartView1)

    # set active source
    SetActiveSource(solutionpvd)

    # set active source
    SetActiveSource(plotOverLine1)

    # Properties modified on plotOverLine1Display
    plotOverLine1Display.SeriesVisibility = ['arc_length', 'pressure', 'Sw']
    plotOverLine1Display.SeriesColor = ['arc_length', '0', '0', '0', 'pressure', '0.889998', '0.100008', '0.110002', 'Sw', '0.220005', '0.489998', '0.719997', 'vtkValidPointMask', '0.300008', '0.689998', '0.289998', 'Points_X', '0.6', '0.310002', '0.639994', 'Points_Y', '1', '0.500008', '0', 'Points_Z', '0.650004', '0.340002', '0.160006', 'Points_Magnitude', '0', '0', '0']
    plotOverLine1Display.SeriesPlotCorner = ['Points_Magnitude', '0', 'Points_X', '0', 'Points_Y', '0', 'Points_Z', '0', 'Sw', '0', 'arc_length', '0', 'pressure', '0', 'vtkValidPointMask', '0']
    plotOverLine1Display.SeriesLineStyle = ['Points_Magnitude', '1', 'Points_X', '1', 'Points_Y', '1', 'Points_Z', '1', 'Sw', '1', 'arc_length', '1', 'pressure', '1', 'vtkValidPointMask', '1']
    plotOverLine1Display.SeriesLineThickness = ['Points_Magnitude', '2', 'Points_X', '2', 'Points_Y', '2', 'Points_Z', '2', 'Sw', '2', 'arc_length', '2', 'pressure', '2', 'vtkValidPointMask', '2']
    plotOverLine1Display.SeriesMarkerStyle = ['Points_Magnitude', '0', 'Points_X', '0', 'Points_Y', '0', 'Points_Z', '0', 'Sw', '0', 'arc_length', '0', 'pressure', '0', 'vtkValidPointMask', '0']

    # save data
    # SaveData('/media/wdcdrive/cpp/wings/build/solution/final_time_step.csv', proxy=plotOverLine1)
    SaveData(output_csv_file, proxy=plotOverLine1)
