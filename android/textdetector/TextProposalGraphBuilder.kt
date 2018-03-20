package ru.avangard.tensorflow.textdetector

import ru.avangard.tensorflow.utils.argmaxBy
import ru.avangard.tensorflow.utils.keep
import java.lang.Math.max
import java.lang.Math.min
import java.util.ArrayList

/**
 * Created by rimidalv on 3/10/18.
 */
class TextProposalGraphBuilder {
    lateinit var textProposals: List<Box>
    lateinit var heights: List<Float>
    lateinit var boxesTable: List<ArrayList<Int>>
    var imgSize: Int = 0

    fun getSuccessions(index: Int): ArrayList<Int> {
        val box = textProposals[index]
        val results = arrayListOf<Int>()
        val from = (box.x1 + 1).toInt()
        val to = min((box.x1 + TextDetector.MAX_HORIZONTAL_GAP + 1).toInt(), imgSize)

        for (left in from until to) {
            val adjBoxIndices = boxesTable[left]
            for (adjBoxIndex in adjBoxIndices) {
                if (meet_v_iou(adjBoxIndex, index))
                    results += adjBoxIndex
            }
            if (results.size != 0)
                return results
        }
        return results
    }

    fun getPrecursors(index: Int): ArrayList<Int> {
        val box = textProposals[index]
        val results = arrayListOf<Int>()
        val from = (box.x1 - 1).toInt()
        val to = max((box.x1 - TextDetector.MAX_HORIZONTAL_GAP).toInt(), 0) - 1

        for (left in from downTo to) {
            val adjBoxIndices = boxesTable[left]
            for (adjBoxIndex in adjBoxIndices) {
                if (meet_v_iou(adjBoxIndex, index))
                    results += adjBoxIndex
            }
            if (results.size != 0)
                return results
        }
        return results
    }

    fun isSuccessionNode(index: Int, succession_index: Int): Boolean {
        val precursors = getPrecursors(succession_index)
        if (textProposals[index].s >= textProposals.keep(precursors).maxBy { it.s }?.s!!)
            return true
        return false
    }

    fun meet_v_iou(index1: Int, index2: Int): Boolean {
        fun overlaps_v(index1: Int, index2: Int): Float {
            val h1 = heights[index1]
            val h2 = heights[index2]
            val y0 = max(textProposals[index2].y1, textProposals[index1].y1)
            val y1 = min(textProposals[index2].y2, textProposals[index1].y2)
            return max(0.0f, y1 - y0 + 1) / min(h1, h2)
        }

        fun size_similarity(index1: Int, index2: Int): Float {
            val h1 = heights[index1]
            val h2 = heights[index2]
            return min(h1, h2) / max(h1, h2)
        }
        return overlaps_v(index1, index2) >= TextDetector.MIN_V_OVERLAPS &&
                size_similarity(index1, index2) >= TextDetector.MIN_SIZE_SIM
    }

    fun buildGraph(text_proposals: List<Box>, imgSize: Int): Graph {
        this.textProposals = text_proposals
//        this.scores = scores
        this.imgSize = imgSize
        this.heights = text_proposals.map { it.y2 - it.y1 + 1 }
//        this.heights = text_proposals[:, 3]-text_proposals[:, 1]+1

        boxesTable = (0 until imgSize).map { arrayListOf<Int>() }

        textProposals.forEachIndexed { index, box ->
            boxesTable[box.x1.toInt()] += index
        }
        val graph = (0 until text_proposals.size).map { Array<Boolean>(text_proposals.size, init = {false}) }
//        graph = np.zeros((text_proposals.shape[0], text_proposals.shape[0]), np.bool)

        textProposals.forEachIndexed { index, box ->
            val successions = getSuccessions(index)
            if (successions.size != 0) {
//                successions.maxBy {  }
                val keep = textProposals.keep(successions)
                val argmaxBy = keep.argmaxBy { it.s }
                val succession_index = successions[argmaxBy]
                if (isSuccessionNode(index, succession_index))
//        # NOTE: a box can have multiple successions(precursors) if multiple successions(precursors)
//        # have equal scores.
                    graph[index][succession_index] = true
            }
        }
        return Graph(graph)
    }
}