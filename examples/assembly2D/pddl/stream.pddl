(define (stream construction)
;    (:stream sample-move
;     ;;; t1 is the target print traj, we need its attachment info
;     :inputs (?r ?q1 ?q2 ?t1)
;     :domain (and (Robot ?r)
;                  (Conf ?r ?q1)
;                  (Conf ?r ?q2)
;                  (Traj ?r ?t1)
;                  (AtStart ?q2 ?t1)
;                  )
;     ; :fluents (Assembled)
;     :outputs (?t2)
;     :certified (and
;                   (MoveAction ?r ?q1 ?q2 ?t2)
;                   (Traj ?r ?t2)
;                   )
;   )

  (:stream sample-place
    :inputs (?r ?e)
    :domain (and (Robot ?r) (Element ?e) (Assigned ?r ?e))
    ; :fluents (Assembled)
    :outputs (?q1 ?q2 ?t)
    :certified (and
                    (PlaceAction ?r ?e ?q1 ?q2 ?t)
                    (Conf ?r ?q1)
                    (Conf ?r ?q2)
                    (Traj ?r ?t)
                    (AtStart ?q1 ?t)
                )
  )

  (:stream test-cfree
    :inputs (?r ?t ?e)
    :domain (and (Robot ?r) (Traj ?r ?t) (Element ?e))
    :certified (CollisionFree ?r ?t ?e)
  )

;   (:stream test-stiffness
; ;    :fluents (Printed)
;    :certified (Stiff)
;   )
)
