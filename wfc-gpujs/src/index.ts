import { GPU } from "gpu.js";
import { real_main_with_make_propergate_fn } from "../../pkg/waveform_function_collapse_lib";

const gpu = new GPU({
    mode: "webgl"
});

export function makePropergateFn(sourceMapRows: number, sourceMapCols: number, targetMapRows: number, targetMapColumns: number, numUniqueTiles: number): (sourceMap: number[][], targetMap: number[][][]) => number[][][] {
    const propergate = gpu.createKernel(function(sourceMap: number[][], targetMap: number[][][]) {
        let targetRow = this.thread.y;
        let targetCol = this.thread.z;
        let targetTileIndex = this.thread.x;
        let targetVal = targetMap[targetRow][targetCol][targetTileIndex];
        let result = 0.0;
        if (targetVal != 0.0) {
            for (let sourceRow = 0; sourceRow < (this.constants.sourceMapRows as number); sourceRow++) {
                for (let sourceCol = 0; sourceCol < (this.constants.sourceMapCols as number); sourceCol++) {
                    let sourceVal = sourceMap[sourceRow][sourceCol];
                    if (sourceVal == targetTileIndex) {
                        let match = 0;
                        if (sourceRow > 0) {
                            let sourceUp = sourceMap[sourceRow - 1][sourceCol];
                            if (targetRow > 0) {
                                if (targetMap[targetRow - 1][targetCol][sourceUp] != 0.0) {
                                    match += 1;
                                }
                            } else {
                                match += 1;
                            }
                        }
                        if (sourceRow < (this.constants.sourceMapRows as number) - 1) {
                            let sourceDown = sourceMap[sourceRow + 1][sourceCol];
                            if (targetRow < (this.constants.targetMapRows as number) - 1) {
                                if (targetMap[targetRow + 1][targetCol][sourceDown] != 0.0) {
                                    match += 1;
                                }
                            } else {
                                match += 1;
                            }
                        }
                        if (sourceCol > 0) {
                            let sourceLeft = sourceMap[sourceRow][sourceCol - 1];
                            if (targetCol > 0) {
                                if (targetMap[targetRow][targetCol - 1][sourceLeft] != 0.0) {
                                    match += 1;
                                }
                            } else {
                                match += 1;
                            }
                        }
                        if (sourceCol < (this.constants.sourceMapCols as number) - 1) {
                            let sourceRight = sourceMap[sourceRow][sourceCol + 1];
                            if (targetCol < (this.constants.targetMapColumns as number) - 1) {
                                if (targetMap[targetRow][targetCol + 1][sourceRight] != 0.0) {
                                    match += 1;
                                }
                            } else {
                                match += 1;
                            }
                        }
                        if (match == 4) {
                            result += 1.0;
                        }
                    }
                }
            }
        }
        return result;
    }, {
        output: [numUniqueTiles, targetMapColumns, targetMapRows],
        constants: {
            sourceMapRows,
            sourceMapCols,
            targetMapRows,
            targetMapColumns,
            numUniqueTiles,
        },
    });
    return (sourceMap: number[][], targetMap: number[][][]) => propergate(sourceMap, targetMap) as number[][][];
}

real_main_with_make_propergate_fn(makePropergateFn);
