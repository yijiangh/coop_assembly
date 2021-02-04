(define (stream construction)
   ; we can try the original incremental
   ; by commenting out the fluent
   ; uncomment the test-cfree stream function
   ; and (in the domain_fluent file) nSafeTraj in the place action's precondition
  (:stream sample-remove
    :inputs (?r ?e)
    :domain (and
               (Robot ?r) (Element ?e) (Assigned ?r ?e)
               )
    :fluents (Assembled)
    :outputs (?q1 ?t ?q2 ?g)
    :certified (and
                  (RemoveMovement ?r ?q1 ?t ?q2 ?e ?g)
                  (Conf ?r ?q1)
                  (Conf ?r ?q2)
                  (Grasp ?e ?g)
                )
  )

   (:stream sample-holding-motion
    :inputs (?r ?q1 ?q2 ?e ?g)
    :domain (and (Robot ?r)
                 (Conf ?r ?q1)
                 (Conf ?r ?q2)
                 (Grasp ?e ?g)
                )
    :fluents (Assembled)
    :outputs (?t)
    :certified (and
                  (HoldingMovement ?r ?q1 ?t ?q2 ?e ?g)
                )
  )

   (:stream sample-free-motion
    ; we need the place traj `pt` for the attachment info
    :inputs (?r ?q1 ?q2)
    :domain (and (Robot ?r)
                 (Conf ?r ?q1)
                 (Conf ?r ?q2)
                )
    :fluents (Assembled)
    :outputs (?t)
    :certified (and
                  (FreeMovement ?r ?q1 ?t ?q2)
                )
  )
)
