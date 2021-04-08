# Copyright (C) 2021 Ikomia SAS
# Contact: https://www.ikomia.com
#
# This file is part of the IkomiaStudio software.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from ikomia import utils, core, dataprocess
import YoloV5Train_process as processMod

#PyQt GUI framework
from PyQt5.QtWidgets import *


# --------------------
# - Class which implements widget associated with the process
# - Inherits PyCore.CProtocolTaskWidget from Ikomia API
# --------------------
class YoloV5TrainWidget(core.CProtocolTaskWidget):

    def __init__(self, param, parent):
        core.CProtocolTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = processMod.YoloV5TrainParam()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.grid_layout = QGridLayout()

        # Dataset folder
        self.browse_dataset_folder = utils.append_browse_file(self.grid_layout, label="Dataset folder",
                                                              path=self.parameters.dataset_folder,
                                                              tooltip="Select folder",
                                                              mode=QFileDialog.Directory)

        # Model name
        self.combo_model_name = utils.append_combo(self.grid_layout, "Model name")
        self.combo_model_name.addItem("yolov5s")
        self.combo_model_name.addItem("yolov5m")
        self.combo_model_name.addItem("yolov5l")
        self.combo_model_name.addItem("yolov5x")
        self.combo_model_name.setCurrentText(self.parameters.model_name)

        # Epochs
        self.spin_epochs = utils.append_spin(self.grid_layout, "Epochs", self.parameters.epochs)

        # Batch size
        self.spin_batch = utils.append_spin(self.grid_layout, "Batch size", self.parameters.batch_size)

        # Input size
        self.spin_input_w = utils.append_spin(self.grid_layout, "Input width", self.parameters.input_size[0])
        self.spin_input_h = utils.append_spin(self.grid_layout, "Input height", self.parameters.input_size[1])

        # Hyper-parameters
        custom_hyp = bool(self.parameters.custom_hyp_file)
        self.check_hyp = QCheckBox("Custom hyper-parameters")
        self.check_hyp.setChecked(custom_hyp)
        self.grid_layout.addWidget(self.check_hyp, self.grid_layout.rowCount(), 0, 1, 2)
        self.check_hyp.stateChanged.connect(self.on_custom_hyp_changed)

        self.label_hyp = QLabel("Hyper-parameters file")
        self.browse_hyp_file = utils.BrowseFileWidget(path=self.parameters.custom_hyp_file,
                                                      tooltip="Select file",
                                                      mode=QFileDialog.ExistingFile)

        row = self.grid_layout.rowCount()
        self.grid_layout.addWidget(self.label_hyp, row, 0)
        self.grid_layout.addWidget(self.browse_hyp_file, row, 1)

        self.label_hyp.setVisible(custom_hyp)
        self.browse_hyp_file.setVisible(custom_hyp)

        # Output folder
        self.browse_out_folder = utils.append_browse_file(self.grid_layout, label="Output folder",
                                                          path=self.parameters.output_folder,
                                                          tooltip="Select folder",
                                                          mode=QFileDialog.Directory)

        # PyQt -> Qt wrapping
        layout_ptr = utils.PyQtToQt(self.grid_layout)

        # Set widget layout
        self.setLayout(layout_ptr)

    def on_custom_hyp_changed(self, int):
        self.label_hyp.setVisible(self.check_hyp.isChecked())
        self.browse_hyp_file.setVisible(self.check_hyp.isChecked())

    def onApply(self):
        # Apply button clicked slot
        # Get parameters from widget
        self.parameters.dataset_folder = self.browse_dataset_folder.path
        self.parameters.model_name = self.combo_model_name.currentText()
        self.parameters.epochs = self.spin_epochs.value()
        self.parameters.batch_size = self.spin_batch.value()
        self.parameters.input_size = [self.spin_input_w.value(), self.spin_input_h.value()]

        if self.check_hyp.isChecked():
            self.parameters.custom_hyp_file = self.browse_hyp_file.path

        self.parameters.output_folder = self.browse_out_folder.path

        # Send signal to launch the process
        self.emitApply(self.parameters)


# --------------------
# - Factory class to build process widget object
# - Inherits PyDataProcess.CWidgetFactory from Ikomia API
# --------------------
class YoloV5TrainWidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the name of the process -> it must be the same as the one declared in the process factory class
        self.name = "YoloV5Train"

    def create(self, param):
        # Create widget object
        return YoloV5TrainWidget(param, None)
