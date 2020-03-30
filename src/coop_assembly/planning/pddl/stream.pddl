(define (stream construction)
   (:stream sample-move
    :inputs (?r ?q2)
    :domain (and (Robot ?r)
                 (Conf ?r ?q2)
                 )
    :outputs (?t)
    :certified (and (MoveAction ?r ?q2 ?t)
                    ; (MoveAction ?r ?q2 ?q1 ?t)
                    (Traj ?t)
                    )
  )

  (:stream sample-print
    :inputs (?r ?e)
    :domain (and (Robot ?r) (Element ?e))
    ; :fluents (Printed)
    :outputs (?q1 ?q2 ?t)
    :certified (and (PrintAction ?r ?e ?q1 ?q2 ?t)
                    (Conf ?r ?q1)
                    (Conf ?r ?q2)
                    (Traj ?t)
                    )
  )

  (:stream test-stiffness
;    :fluents (Printed)
   :certified (Stiff)
  )
)
