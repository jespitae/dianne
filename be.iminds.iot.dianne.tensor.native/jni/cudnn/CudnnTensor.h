/*******************************************************************************
 * DIANNE  - Framework for distributed artificial neural networks
 * Copyright (C) 2015  iMinds - IBCN - UGent
 *
 * This file is part of DIANNE.
 *
 * DIANNE is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * Contributors:
 *     Tim Verbelen, Steven Bohez, Elias De Coninck
 *******************************************************************************/
#ifndef CUDNNTENSOR_H
#define CUDNNTENSOR_H

#include <cudnn.h>

#define checkCUDNN(status) do {                                        \
    if (status != CUDNN_STATUS_SUCCESS) {                              \
      printf("%s %s:%d \n", cudnnGetErrorString(status), __FILE__, __LINE__); \
      throwException(cudnnGetErrorString(status));  			       \
    }                                                                  \
} while(0)


extern cudnnHandle_t cudnnHandle;

extern int convFwAlg;
extern int convBwAlg;
extern int convAgAlg;

extern int workspaceLimit;
extern int shareWorkspace;
extern size_t workspaceSize;
extern void* workspace;


#endif
