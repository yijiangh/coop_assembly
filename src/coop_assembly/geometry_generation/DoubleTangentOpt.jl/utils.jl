# ported from pddlstream.utils
# https://github.com/caelan/pddlstream/blob/stable/pddlstream/utils.py

"""
find an unique element in a sequence satisfying a `test`
"""
function find_unique(test_fn, sequence)
    found, value = false, nothing
    for item in sequence
        if test_fn(item)
            if found
                throw(error("Both elements $value and $item satisfy the test"))
            end
            found, value = true, item
        end
    end
    if !found
        throw(error("Unable to find an element satisfying the test"))
    end
    return value
end

