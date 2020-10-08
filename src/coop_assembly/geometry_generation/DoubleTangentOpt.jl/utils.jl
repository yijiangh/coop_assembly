# ported from pddlstream.utils
# https://github.com/caelan/pddlstream/blob/stable/pddlstream/utils.py


"""
find an unique element in a sequence satisfying a `test`
"""
function find_unique(test, sequence)
    found, value = False, None
    for item in sequence
        if test(item)
            if found
                throw(Error("Both elements $value and $item satisfy the test"))
            found, value = True, item
            end
        end
    end
    if !found
        throw("Unable to find an element satisfying the test")
    end
    return value
end

