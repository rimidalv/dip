package ru.avangard.tensorflow.textdetector

import android.util.Log
import ru.avangard.tensorflow.utils.keep
import java.lang.Math.*
import java.util.*


/**
 * Created by rimidalv on 3/10/18.
 */
class TextDetector(val imgW: Int, val imgH: Int) {

    val graphBuilder = TextProposalGraphBuilder()

    fun detect(proposals: List<Box>): List<FinalBox> {
        val sortedProposals = proposals.filter { it.s > TEXT_PROPOSALS_MIN_SCORE }
//        val sortedProposals = proposals.sortedBy { it.s }
        val keep_inds = nms(sortedProposals, TEXT_PROPOSALS_NMS_THRESH)
        val nmsProposals = sortedProposals.keep(keep_inds)
        val textRecs = getTextLines(nmsProposals)
        val boxes = filterBoxes(textRecs)
        return boxes
    }

    fun filterBoxes(boxes: Array<FinalBox>): List<FinalBox> {
        return boxes.filter { box ->
            val heights = box.h //(abs(box[5] - box[1]) + abs(box[7] - box[3])) / 2.0 + 1;
            val widths = box.w //(abs(box[2] - box[0]) + abs(box[6] - box[4])) / 2.0 + 1;
            val scores = box.s

            (widths / heights > MIN_RATIO) && (scores > LINE_MIN_SCORE) &&
                    (widths > (TEXT_PROPOSALS_WIDTH * MIN_NUM_PROPOSALS))
        }
    }

    fun threshold(coord: Float, min_: Float, max_: Float): Float {
        return max(min(coord, max_), min_)
    }

    fun clipBoxes(boxes: Array<Box?>): List<Box> {
        val notNull = boxes.filterNotNull()
        notNull.forEach {
            it.x1 = threshold(it.x1, 0.0f, (imgW - 1).toFloat())
            it.x2 = threshold(it.x2, 0.0f, (imgW - 1).toFloat())
            it.y1 = threshold(it.y1, 0.0f, (imgH - 1).toFloat())
            it.y2 = threshold(it.y2, 0.0f, (imgH - 1).toFloat())
        }
        return notNull
    }

    fun nms(dets: List<Box>, thresh: Double): ArrayList<Int> {
        val ndets = dets.size
        val suppressed = Array<Int>(ndets, { 0 })
        var ix1 = 0.0f
        var iy1 = 0.0f
        var ix2 = 0.0f
        var iy2 = 0.0f
        var iarea = 0.0f
        var aarea = 0.0f
        var xx1 = 0.0f
        var yy1 = 0.0f
        var xx2 = 0.0f
        var yy2 = 0.0f
        var w = 0.0f
        var h = 0.0f
        var inter = 0.0f
        var over = 0.0f


        val keep = arrayListOf<Int>()
        for (i in 0 until ndets) {
            if (suppressed[i] == 1)
                continue
            keep += i
            ix1 = dets[i].x1
            iy1 = dets[i].y1
            ix2 = dets[i].x2
            iy2 = dets[i].y2
            iarea = dets[i].area

            for (j in (i + 1) until ndets) {
                if (suppressed[j] == 1)
                    continue

                xx1 = max(ix1, dets[j].x1)
                yy1 = max(iy1, dets[j].y1)
                xx2 = min(ix2, dets[j].x2)
                yy2 = min(iy2, dets[j].y2)
                aarea = dets[j].area
                w = max(0.0f, xx2 - xx1 + 1)
                h = max(0.0f, yy2 - yy1 + 1)
                inter = w * h
                over = inter / (iarea + aarea - inter)
                if (over >= thresh)
                    suppressed[j] = 1
            }
        }
        return keep
    }

    fun groupTextProposals(textProposals: List<Box>): ArrayList<ArrayList<Int>> {
        val graph = graphBuilder.buildGraph(textProposals, imgW)
        return graph.subGraphsConnected()
    }

    //def group_text_proposals(self, textProposals, scores, im_size):
//    graph=self.graph_builder.build_graph(textProposals, scores, im_size)
//    return graph.sub_graphs_connected()
//

    fun fitY(X: List<Float>, Y: List<Float>, x1: Float, x2: Float): Pair<Float, Float> {
        if (X.count { X.all { it == X[0] } } == X.size) {
            return Pair(Y[0], Y[0])
        }
//        val p = polyRegression(X, Y)
        val p = polyFit(X, Y, X.size, 1)
//        Log.d("PPPPPPP", Arrays.toString(p))
//        return Pair(p(x1), p(x2))
        val a1 = (p[1] * x1 + p[0]).toFloat()
        val a2 = (p[1] * x2 + p[0]).toFloat()
        return Pair(a1, a2)
    }

    fun polyFit(x: List<Float>, y: List<Float>, N: Int, n1: Int): DoubleArray { //(Float) -> Unit{
        var n = n1
        val X = DoubleArray(2 * n + 1)
        for (i in 0 until 2 * n + 1) {
            X[i] = 0.0
            for (j in 0 until N)
                X[i] = X[i] + Math.pow(x[j].toDouble(), i.toDouble())        //consecutive positions of the array will store N,sigma(xi),sigma(xi^2),sigma(xi^3)....sigma(xi^2n)
        }
        val B = Array(n + 1) { DoubleArray(n + 2) }
        val a = DoubleArray(n + 1)            //B is the Normal matrix(augmented) that will store the equations, 'a' is for value of the final coefficients
        for (i in 0..n)
            for (j in 0..n)
                B[i][j] = X[i + j]            //Build the Normal matrix by storing the corresponding coefficients at the right positions except the last column of the matrix
        val Y = DoubleArray(n + 1)                    //Array to store the values of sigma(yi),sigma(xi*yi),sigma(xi^2*yi)...sigma(xi^n*yi)
        for (i in 0 until n + 1) {
            Y[i] = 0.0
            for (j in 0 until N)
                Y[i] = Y[i] + Math.pow(x[j].toDouble(), i.toDouble()) * y[j]        //consecutive positions will store sigma(yi),sigma(xi*yi),sigma(xi^2*yi)...sigma(xi^n*yi)
        }
        for (i in 0..n)
            B[i][n + 1] = Y[i]                //load the values of Y as the last column of B(Normal Matrix but augmented)
        n = n + 1
        for (i in 0 until n)
        //From now Gaussian Elimination starts(can be ignored) to solve the set of linear equations (Pivotisation)
            for (k in i + 1 until n)
                if (B[i][i] < B[k][i])
                    for (j in 0..n) {
                        val temp = B[i][j]
                        B[i][j] = B[k][j]
                        B[k][j] = temp
                    }

        for (i in 0 until n - 1)
        //loop to perform the gauss elimination
            for (k in i + 1 until n) {
                val t = B[k][i] / B[i][i]
                for (j in 0..n)
                    B[k][j] = B[k][j] - t * B[i][j]    //make the elements below the pivot elements equal to zero or elimnate the variables
            }
        for (i in n - 1 downTo 0)
        //back-substitution
        {                        //x is an array whose values correspond to the values of x,y,z..
            a[i] = B[i][n]                //make the variable to be calculated equal to the rhs of the last equation
            for (j in 0 until n)
                if (j != i)
                //then subtract all the lhs values except the coefficient of the variable whose value                                   is being calculated
                    a[i] = a[i] - B[i][j] * a[j]
            a[i] = a[i] / B[i][i]            //now finally divide the rhs by the coefficient of the variable to be calculated
        }

        return a
    }


    fun getTextLines(textProposals: List<Box>): Array<FinalBox> {
        val tpGroups = groupTextProposals(textProposals)
        val textLines = Array<Box?>(tpGroups.size, init = { null })
        tpGroups.forEachIndexed { index, tpIndices ->
            val textLineBoxes = textProposals.keep(tpIndices)
            val x1 = textLineBoxes.minBy { it.x1 }!!.x1
            val x2 = textLineBoxes.maxBy { it.x2 }!!.x2

            val offset = ((textLineBoxes[0].x2 - textLineBoxes[0].x1) * 0.5).toFloat()
            val (lt_y, rt_y) = fitY(textLineBoxes.map { it.x1 }, textLineBoxes.map { it.y1 }, x1 + offset, x2 - offset)
            val (lb_y, rb_y) = fitY(textLineBoxes.map { it.x1 }, textLineBoxes.map { it.y2 }, x1 + offset, x2 - offset)

            val score = (textLineBoxes.sumByDouble { it.s.toDouble() } / tpIndices.size).toFloat()

            val y1 = min(lt_y, rt_y)
            val y2 = max(lb_y, rb_y)
            val box = Box(
                    x1 = x1,
                    y1 = y1,
                    x2 = x2,
                    y2 = y2,
                    s = score)
            textLines[index] = box
        }
        val textLinesClipped = clipBoxes(textLines)

        val textRecs = Array(textLinesClipped.size, init = { FinalBox() })
        var index = 0
        for (line in textLinesClipped) {
            val xmin = line.x1
            val ymin = line.y1
            val xmax = line.x2
            val ymax = line.y2
//            textRecs[index][0] = xmin
//            textRecs[index][1] = ymin
//            textRecs[index][2] = xmax
//            textRecs[index][3] = ymin
//            textRecs[index][4] = xmin
//            textRecs[index][5] = ymax
//            textRecs[index][6] = xmax
//            textRecs[index][7] = ymax

            val heights = abs(ymax - ymin) + 1
            val widths = abs(xmax - xmin) + 1

            textRecs[index].x = xmin
            textRecs[index].y = ymin
            textRecs[index].w = widths
            textRecs[index].h = heights

            textRecs[index].s = line.s

            index = index + 1
        }
        return textRecs
    }


    companion object {
        val SCALE = 400
        val MAX_SCALE = 800
        val TEXT_PROPOSALS_WIDTH = 16
        val MIN_NUM_PROPOSALS = 2
        val MIN_RATIO = 0.5
        val LINE_MIN_SCORE = 0.9
        val MAX_HORIZONTAL_GAP = 50
        val TEXT_PROPOSALS_MIN_SCORE = 0.7
        val TEXT_PROPOSALS_NMS_THRESH = 0.2
        val MIN_V_OVERLAPS = 0.7
        val MIN_SIZE_SIM = 0.7
    }
}