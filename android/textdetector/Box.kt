package ru.avangard.tensorflow.textdetector

/**
 * Created by rimidalv on 3/10/18.
 */
data class Box(val s: Float, var x1: Float, var y1: Float, var x2: Float, var y2: Float){
    var area = (x2 - x1 + 1) * (y2 - y1 + 1)
}